# Score-matching neural networks for likelihood based galaxy morphology priors
## Matt Sampson

<img src="/images/compare.png" height="400">

For the use of galaxy deblending we construct a time-independent 
score matching model (diffusion model) with a U-NET architecture. The architecture
is a JAX -> equinox port from Song+21 NCSN (https://arxiv.org/abs/2011.13456  ).

This score network is to be used as a galaxy morphology prior in the upcoming 
SCARLET 2 details: https://pmelchior.net/blog/scarlet2-redesign.html. Scarlet 2 code here: https://github.com/pmelchior/scarlet2

## Details:
The neural network architecture "ScoreNet" is located in models_eqx.py
The loss function is defined in the training script train_script.py which takes in command
line inputs for the image size to train on. 
Weights and bias's are saved as PyTrees which can then be read in and loaded (see here: https://docs.kidger.site/equinox/api/utilities/serialisation/). 

## Current results and functionality
ScoreNet works well at removing visual artifacts such as ring patterns or multiple sources. An high score in absolute values indicates a pixel or pixel region is not consistent with the prior distribution p(x) (the training data) which here is large samples of single source SCARLET models from the Subaru Hyper SUprime-Cam.
<img src="/images/score_runtests.png" height="400">


<img src="/images/HSC_res64_artifact2.jpg" height="200"> <img src="/images/rings_single.gif" width="200" height="200"/> <img src="/images/HSC_res64_artifact.jpg" height="200"> <img src="/images/multi.gif" width="200" height="200"/>





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
