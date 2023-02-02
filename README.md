# Score-matching neural networks for galaxy morphology priors
<img src="/images/scorenet_decal.jpg" height="300">   

## Matt Sampson

<img src="/images/compare.png" height="400">

For the use of galaxy deblending we construct a time-independent 
score matching model (diffusion model) with a U-NET architecture. The architecture
is a JAX -> equinox port from Song+21 NCSN (https://arxiv.org/abs/2011.13456) mostly by (https://docs.kidger.site/equinox/examples/score_based_diffusion/) with minor changes.

This score network is to be used as a galaxy morphology prior in the upcoming 
SCARLET 2 details: https://pmelchior.net/blog/scarlet2-redesign.html. Scarlet 2 code here: https://github.com/pmelchior/scarlet2

## Details:
The neural network architecture "ScoreNet" is located in models_eqx.py
The loss function is defined in the training script train_script.py that takes in command
line inputs for the image size to train on. 
Weights and biases are saved as PyTrees which can then be read in and loaded (see here: https://docs.kidger.site/equinox/api/utilities/serialisation/). 

## Current results and functionality
ScoreNet works well at removing visual artifacts such as ring patterns and multiple sources. A high score in absolute values indicates a pixel or pixel region is not consistent with the prior distribution p(x) (the training data) which here is large samples of single source SCARLET models from the Subaru Hyper Suprime-Cam.
<img src="/images/score_runtests.png" height="400">


<img src="/images/HSC_res64_artifact2.jpg" height="200"> <img src="/images/rings_single.gif" width="200" height="200"/> <img src="/images/HSC_res64_artifact.jpg" height="200"> <img src="/images/multi.gif" width="200" height="200"/>

## Useage -- see demo notebook
Simple case to return the score function of an image. Currently works on 32 by 32 and 64 by 64 size images but can be altered. Note the input image must be in dimensions (n, size, size) where n is the number of images, n is the image size either 32 or 64. If using (size, size) data apply jnp.expand_dims(data, axis=0) to add an arbitrary first dimension. 

To load score networks:

from scorenet import ScoreNet32, ScoreNet64

y = image_32  
score32 = ScoreNet32(y)  
print(f'size score is {score32.shape}')  
score should be a (1, 32, 32) size array  
plt.imshow(score32[0])  
plt.show()  

y = image_64  
score64 = ScoreNet64(y)  
print(f'size score is {score64.shape}')  
score should be a (1, 64, 64) size array  
plt.imshow(score64[0])  
plt.show()  

Optional, note that ScoreNetXX(y,t=0). ScoreNet takes in an image and can optionally be passes a time. This time corresponds to a noise level of the input image where t = 10 means the image will be blurred to close to pure Gaussian noise then the score is taken. Setting t = 0, which is defaul means no noise will be added. The noise level scales from max at t = 10 (arbitrarily chosen at training) to t = 0 via a scaling image = image + noise * (1 - exp(-t))



## Useful papers
### For context scientific context:

Scarlet paper: (https://ui.adsabs.harvard.edu/abs/2018A&C....24..129M)

### Similar work:

Song+2019 and 2020 (https://arxiv.org/abs/1907.05600 , https://arxiv.org/abs/2006.09011)

Burke+2019 (https://arxiv.org/abs/1908.02748)

Huang+2022 (https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)

## Useful other rescources
Lilian Weng blogpost (https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

Yang Song blogpost (https://yang-song.github.io/blog/2021/score/)

For details/issues/plot aesthetic suggestions
email: matt.sampson@princeton.edu 
