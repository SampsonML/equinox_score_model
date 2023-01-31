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


def main(
    # Model hyperparameters
    patch_size=4,
    hidden_size=64,
    mix_patch_size=512,
    mix_hidden_size=512,
    num_blocks=4,
    t1=10.0,
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
    
    key = jr.PRNGKey(42)
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

    # load stored model
    SAVE_DIR = 'stored_models'
    fn = SAVE_DIR + '/eqx_model_step_0_res_64.eqx'
    #eqx.tree_serialise_leaves(fn, model)
    best_model = eqx.tree_deserialise_leaves(fn, model)
    PLOT_DIR = 'plots'
    
    # ----------------- testing ----------------- #
    # testing plot
    test_size = 1
    t = jr.uniform(train_key, (test_size,), minval=0, maxval=0 / test_size)
    t = t + (t1 / test_size) * jnp.arange(test_size)
    print(f'shape of t: {t.shape}')
    print(f't is: {t}')
    y = data[0]
    #y = jnp.squeeze(y,axis=0)
    print(f'shape of y: {y.shape}')
    score = best_model(t,y)
    print(f'shape of score: {score.shape}')
    fig = plt.figure(figsize=(16, 16), dpi = 250)
    plt.subplot(1,2,1)
    plt.imshow(score,cmap='plasma')
    plt.subplot(1,2,2)
    plt.imshow(y)
    plt.savefig(PLOT_DIR + '/score_test.png')
    # ------------------- end ------------------- #

    sample_key = jr.split(sample_key, sample_size**2)
    sample_fn = ft.partial(single_sample_fn, best_model, int_beta, data_shape, dt0, t1)
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
    filename = PLOT_DIR + '/galaxies_loaded_' + str(args.size) + '.png'
    plt.savefig(filename,facecolor='black', transparent=False ,dpi = 250)
    filename = 'score_based_equinox_models_res' + str(args.size) + '.pdf'
    plt.savefig(filename,facecolor='black', transparent=False ,dpi = 250)   
    plt.show()

    """
    vis_steps = 20
    t_vec = jnp.linspace(t1, 0, vis_steps)
    for i in range(len(t_vec)):
        t0 = t_vec[i]
        model_key, train_key, loader_key, sample_key = jr.split(key, 4)
        sample_key = jr.split(sample_key, sample_size**2)
        sample_fn = ft.partial(single_sample_fn, best_model, int_beta, data_shape, dt0, t1,t0)
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
        filename = PLOT_DIR + '/galaxies_t_' + str(len(t_vec) - i) + 'res' + str(args.size) + '.png'
        plt.savefig(filename,facecolor='black', transparent=False ,dpi = 250)
        plt.close()
        #filename = PLOT_DIR + '/galaxies_t_' + str(len(t_vec) - i) + 'res' + str(args.size) + '.pdf'
        #plt.savefig(filename,facecolor='black', transparent=False ,dpi = 250)   
    """

# Code entry point
if __name__ == '__main__':
    main()

