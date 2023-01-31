#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------ #
# A jax implementation of the diffusion model from                               #
# the paper:                                                                     #
# "Score-Based Generative Modeling through Stochastic Differential Equations"    #
# https://arxiv.org/abs/2011.13456                                               #
# Code taken primarily from https://github.com/yang-song/score_sde/              #
# # and https://docs.kidger.site/equinox/examples/score_based_diffusion/         #
# Author: Matt Sampson                                                           #
# Created: 2023                                                                  #
# ------------------------------------------------------------------------------ #

import os
import functools as ft
import diffrax as dfx  # https://github.com/patrick-kidger/diffrax
import einops  # https://github.com/arogozhnikov/einops
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import numpy as np

import equinox as eqx
import cmasher as cmr
import argparse
from models_eqx import ScoreNet

print('    _____                         _   _        _   ')
print('   /  ___|                       | \ | |      | |  ')
print('   \ `--.   ___  ___   _ __  ___ |  \| |  ___ | |_  ')
print('    `--. \ / __|/ _ \ |  __|/ _ \| . ` | / _ \| __|')
print('   /\__/ /| (__| (_) || |  |  __/| |\  ||  __/| |_ ')
print('   \____/  \___|\___/ |_|   \___|\_| \_/ \___| \__|')
print('   Generating galaxies from noise with deep learning')     
print('                  <>  Matt Sampson  <>')                                       

#print(f'Device used: {xla_bridge.get_backend().platform}')

# parse in the image size to train on from the command line
# Parse arguements
parser = argparse.ArgumentParser(
    description="training script")
parser.add_argument("-s", "--size",
                    help="size of image to train on",
                    default="32", type=int)
args    = parser.parse_args()

def single_loss_fn(model, weight, int_beta, data, t, key):
    mean = data * jnp.exp(-0.5 * int_beta(t))
    var = jnp.maximum(1 - jnp.exp(-int_beta(t)), 1e-5)
    std = jnp.sqrt(var)
    noise = jr.normal(key, data.shape)
    y = mean + std * noise
    pred = model(t, y)
    return weight(t) * jnp.mean((pred + noise / std) ** 2)


def batch_loss_fn(model, weight, int_beta, data, t1, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, model, weight, int_beta)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(data, t, losskey))


@eqx.filter_jit
def single_sample_fn(model, int_beta, data_shape, dt0, t1, key):
    def drift(t, y, args):
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(t, y))

    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 0
    y1 = jr.normal(key, data_shape)
    # reverse time, solve from t1 to t0
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1, adjoint=dfx.NoAdjoint())
    return sol.ys[0]


def HST_data(im_size):
    if im_size == 32:
        box_size = 31
        dataname = 'sources_box' + str(box_size) + '.npy'     
        dataset = np.load(dataname)
        # perform zero-padding of the data to get desired dimensions
        data_padded_31 = []
        for i in range(len(dataset)):
            data_padded_tmp = np.pad(dataset[i], ((0,1),(0,1)), 'constant')
            data_padded_31.append(data_padded_tmp)
        dataset = np.array( data_padded_31 )

    elif im_size == 64:
        box_size = 51
        dataname = 'sources_box' + str(box_size) + '.npy'     
        dataset = np.load(dataname)
        # perform zero-padding of the data to get desired dimensions
        data_padded_51 = []
        for i in range(len(dataset)):
            data_padded_tmp = np.pad(dataset[i], ((6,7),(6,7)), 'constant')
            data_padded_51.append(data_padded_tmp)
        dataset_51 = np.array( data_padded_51 )
        # load in data  high res
        box_size = 61
        dataname = 'sources_box' + str(box_size) + '.npy'     
        dataset = np.load(dataname)
        # perform zero-padding of the data to get desired dimensions
        data_padded_61 = []
        for i in range(len(dataset)):
            data_padded_tmp = np.pad(dataset[i], ((1,2),(1,2)), 'constant')
            data_padded_61.append(data_padded_tmp)
        # add a loop to add 51 and 61 data together
        for i in range(len(dataset_51)):
            data_padded_61.append( dataset_51[i] )
        dataset = np.array( data_padded_61 )

    # add extra dim for channel dimension
    dataset = np.expand_dims(dataset, axis=1)
    data_jax = jnp.array(dataset)
        
    return data_jax

def dataloader(data, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size


@eqx.filter_jit
def make_step(model, weight, int_beta, data, t1, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, weight, int_beta, data, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state


def main(
    # Model hyperparameters
    patch_size=4,
    hidden_size=64,
    mix_patch_size=512,
    mix_hidden_size=512,
    num_blocks=4,
    t1=10.0,
    # Optimisation hyperparameters
    num_steps=50_000,
    lr=3e-4,
    batch_size=256,
    print_every=5_000,
    # Sampling hyperparameters
    dt0=0.1,
    sample_size=10,
    # Seed
    seed=5678,
):
    
    # save parameters
    SAVE_DIR = 'stored_models'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    key = jr.PRNGKey(seed)
    model_key, train_key, loader_key, sample_key = jr.split(key, 4)
    data = HST_data(args.size)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_max = jnp.max(data)
    data_min = jnp.min(data)
    data_shape = data.shape[1:]
    data = (data - data_mean) / data_std

    model = ScoreNet(
        data_shape,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        key=model_key,
    )
    int_beta = lambda t: t  # Try experimenting with other options here!
    weight = lambda t: 1 - jnp.exp(
        -int_beta(t)
    )  # Just chosen to upweight the region near t=0.

    opt = optax.adabelief(lr)
    # Optax will update the floating-point JAX arrays in the model.
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    total_value = 0
    total_size = 0
    for step, data in zip(
        range(num_steps), dataloader(data, batch_size, key=loader_key)
    ):
        value, model, train_key, opt_state = make_step(
            model, weight, int_beta, data, t1, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1
        if (step % print_every) == 0 or step == num_steps - 1:
            print(f"Step={step} Loss={total_value / total_size}")
            total_value = 0
            total_size = 0
            
            # save the model
            fn = SAVE_DIR + 'eqx_model_step_' +str(step) + '_res_' + str(args.size) + '.eqx'
            eqx.tree_serialise_leaves(fn, model)

    sample_key = jr.split(sample_key, sample_size**2)
    sample_fn = ft.partial(single_sample_fn, model, int_beta, data_shape, dt0, t1)
    sample = jax.vmap(sample_fn)(sample_key)
    sample = data_mean + data_std * sample
    sample = jnp.clip(sample, data_min, data_max)
    sample = einops.rearrange(
        sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
    )
    cmap = cmr.lilac
    fig = plt.figure(figsize=(16, 16), dpi = 250)
    plt.style.use('dark_background')
    title = 'score based generation of equinox models'
    plt.suptitle(title, fontsize = 30)
    plt.imshow(sample, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    filename = 'score_based_equinox_models_res' + str(args.size) + '.png'
    plt.savefig(filename,facecolor='black', transparent=False ,dpi = 250)
    filename = 'score_based_equinox_models_res' + str(args.size) + '.pdf'
    plt.savefig(filename,facecolor='black', transparent=False ,dpi = 250)   
    plt.show()

# Code entry point
if __name__ == '__main__':
    main()

