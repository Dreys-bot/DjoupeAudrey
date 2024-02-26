---
layout: '@/templates/BasePost.astro'
title: 3D Generative Adversial Model with VAE encoders
description: The project aims to implement and compare GAN, VAE-GAN and MV-VAE-GAN models for 3D shape generation. These models are trained to generate new shapes from latent spaces.
githubPage: https://github.com/Dreys-bot/3DVAEGAN/tree/main
pubDate: 2023-12-05T00:00:00Z
iconSrc: '/assets/icons/3dgan-icon.png'
iconAlt: '3D GAN icon'
imgSrc: '/assets/images/paint-icon.png'
imgAlt: 'Paint icon'
tags: ['Python', 'Torch', 'Numpy', 'Matplotlib', 'PIL']
---

# Summary

[[#Abstract]]
[[#Introduction]]
[[#Method]]
	[[#Architectures]]
	[[#Model Parameters]]
[[#Data processing]]
[[#Implementation]]
[[#Results]]


# Abstract

Generative models are a commonly studied topic in artifical intelligence that aim to understand the underlying structure of data by estimating a probability distribution of a dataset. Variational Autoencoders (VAE) and General Adverserial Networks (GAN) have become two of the most popular generative models. Promising results have been shown when combining VAEs and GANs by sharing the decoder network of a VAE and generator network of the GAN. Recent work has extended these architectures into the 3D realm. The 3D-GAN and 3D-VAE-GAN architectures are explored in this project and a new architecture is introduced to incorporate multiple views with the 3D-VAE-GAN for 3D object generation.

# Introduction

The goal of many artificial intelligence researchers is to achieve human level intelligence in machines. Many algorithms are designed to learn and understand a given dataset to perform tasks such as classification. Humans have the ability to create things such as music, art, stories, etc. Thus, in order for machines to achieve human level intelligence, they must be able to create as well as understand. Generative models are a step towards teaching machines how to create. They aim to estimate the probability distribution of a dataset and novel data points could be sampled from this distribution. Currently three main approach to generative models exist in machine learning: General Adverserial Networks (GANs) [2], Variational Autoencoeders (VAEs) [3], and Autoregressive models (such as PixelRNN [12]). GAN architectures use a two player adversial approach, where one model tries to generate realistic looking data and the other model determines which inputs are real and which are generated. A VAE encodes then decodes the input data and learns a latent distribution of the dataset. 

# Method

## Architectures

### 3D-GAN Model


							![[3DGAN.png]]
							***Figure 1. Architecture of GAN. Given a random latent vector $z_{t}$, the generator network G produces a 3D object $x_{t}$. The discriminator network D classifies which object is real and which is fake.***

GANs can be described as a two player game. One player is a generator model which produces fake data points from a latent vector. The second player is a discriminator which is given both the generated data point and a real data point from the dataset and classifies which is real and which is generated. The goal is to have the generator successfully trick the discriminator into classifying the generated data point as real. This would mean the generator is producing data that is realistic. Given the data point $x_{i}$ and a random vector $z_{t}$, the loss of the discriminator tries to maximize $$  
LD = \log D(x_i) + \log(1 - D(G(z_t)))  
$$
and is two-fold so that $$ log D(xi) ∈ (0, 1) $$ is close to 1, and $$ log(1−D(G(zt))) ∈ (0, 1) $$ is close to 0. The generator loss function tries  to minimize $$ LG = 1 − D(G(zt)) $$ so that $$ D(G(Zt)) $$ is close to 1. Then, the total loss during training is combined as $$ L = LD + LG $$
3D-GAN architecture in Figure 1 follows the original GAN architecture, but the output $x_{i}$ is a 3D object instead of an image. It consists of a generator network that repeatedly upsamples with deconvolutional layers, starting from a one dimensional $z_{t}$ vector to a 3D object.

### 3D-VAE-GAN Model


							![[3DVAEGAN.png]]
							***Figure 2. Architecture of 3D-VAE-GAN. The VAE is given a 2D image $y_{i}$ and is encoded by E into a mean $z_{µ}$ and variance $z_{σ}$. A latent vector  is sampled from this distribution and then passed through the generator G to get a 3D object $x_{e}$. A random latent vector $z_{t}$ is passed through G to get the 3D object $x_{t}$. The discriminator D classifies if the real 3D object $x_{i}$ or $x_{t}$ is real or fake. The reconstruction loss is calculated between $x_{e}$ and $x_{i}$.***

The 3D-VAE-GAN proposes an addition to the 3D-GAN architecture. In this variation, the decoder of the VAE is shared with the generator of the GAN seen in Figure 2. Given an image $y_{i}$ and a corresponding 3D object $x_{i}$ , the VAE learns a distribution with $z_{µ}$ and $z_{σ}$. This distribution is sampled to receive $z_{e}$. Then, the generator of the GAN produces a 3D object by upsampling $z_{e}$. During training, reconstructed loss is calculated as the distance between the generated 3D object and input 3D object $$ L_{recon} = ||G(z_{e}) − x_{i}|| $$ 
During training, the KL Divergence is calculated to ensure $z_{µ}$ and $z_{σ}$ are close to a normal distribution. $$ L_{KL} = − 1 2 ∗ \sum(1 + z_{var} − z^2_{µ} − e_{zvar})$$ The loss for the generator is updated to include the reconstruction loss $$ L_{G} = 1 − D(G(z_{t})) + L_{recon} $$ The total loss is the addition of all the separate loss functions $$ L = L_{D} + L_{G} + L_{KL} $$

### MV-3D-VAE-GAN Model
	   
							   ![[MV3DVAEGAN.png]]
							***Figure 3. Architecture of MV-3D-VAE-GAN. The VAE is given multiple 2D images Y . Each image y ∈ Y is encoded by E into their respective mean $z_{µy}$ and variance $z_{σy}$. A latent vector $z_{ey}$ is sampled from each distribution and then all combined using either max pooling or average pooling to receive a single dimension encoding $z_{e}$. Then, $z_{e}$ is passed through the generator G to get a 3D object $x_{e}$. A random latent vector $z_{t}$ is passed through G to get the 3D object $x_{t}$. The discriminator D classifies if the real 3D object $x_{i}$ or $x_{t}$ is real or fake. The reconstruction loss is calculated between $x_{e}$ and $x_{i}$..***



The MV-3D-VAE-GAN is an extension of the 3D-VAEGAN architecture. In this architecture, seen in Figure 3, the goal is to learn a better latent vector of the 2D image by merging multiple 2D images into a single representation. Each view is a 2D image $y_{i}$ ∈ Y of the same 3D object $x_{i}$ and is encoded into a latent vector. Each view learns it’s own distribution $z_{μ_{yi}}$ and $z_{σyi}$ . Each is sampled to receive $z_{eyi}$ and then pooled together to receive a final representation of $z_{e}$. Then, a 3D object is generated and the reconstruction loss is calculated the same way as 3D-VAE-GAN. The only difference during training is the KL divergence is averaged across all views to ensure each $z_{μ_{yi}}$ and $z_{σ_{yi}}$  is close to a normal distribution.

### Small summary
The generator network for each model included five transpose 3D layers with 3D batch normalization and a ReLU activation functions to map a single dimension vector into a 32x32x32 voxelized 3D object. The discriminator network for each model includes five 3D convolutional layers with 3D batch normalization, leaky ReLU activation functions with a sigmoid layer at the end to map a 3D object into a single scalar value representing the probability of being a generated object. The image encoding networks for the 3D-VAE-GAN and MV-3D-VAE-GAN models are a five layer 2D convolution network with 2D batch normalization and ReLU activation functions to map a the 2D image into a single latent dimension. All models where optimized using the ADAM optimizer with a learning rate of 0.0025, 0.0001, 0.001 for the generator, image encoder, and discriminator network.

## Data processing

The 3D-GAN architecture requires only a 3D object while the 3D-VAE-GAN architecture requires a 3D object and a corresponding 2D image of the object. The MV-3DVAE-GAN architecture requires a 3D object and multiple 2D images of the 3D object at different views. The ’modelnet40v1png’ dataset from the MVCNN paper [1] provides all the necessary images and 3D objects for all architectures. For this project, the models are trained using only the chair class from the ModelNet dataset.

The 3D objects were provided in .OFF file format and require a voxelization step as preprocessing. To achieve this, the binvox software developed by Patrick Min [2] was used to convert the .OFF files into .binvox files. Then, during the loading of the dataset in PyTorch, the ’binvox-rw-py’ [3] script by Daniel Maturana is used to convert the binvox files into 3D arrays.

# Implementation

## Model Parameters

In my code, i have defined the main.py to define **arguments** of my training. We have 3 model to train **3DGAN**, **3DVAEGAN**, **3DVAEGAN_MULTIVIEW** and we must specify it when we are launching our algorithm to tell to the programm which type pf model you want to train like in this code.

```python
if args.test == False:
        if args.alg_type == '3DGAN':
            train(args)
        elif args.alg_type == '3DVAEGAN':
            train_vae(args)
        elif args.alg_type == '3DVAEGAN_MULTIVIEW':
            train_multiview(args)
    else:
        if args.alg_type == '3DGAN':
            print("TESTING 3DGAN")
            test_3DGAN(args)
        elif args.alg_type == '3DVAEGAN':
            print("TESTING 3DVAEGAN")
            test_3DVAEGAN(args)
        elif args.alg_type == '3DVAEGAN_MULTIVIEW':
            print("TESTING 3DVAEGANMULTIVIEW")
            test_3DVAEGAN_MULTIVIEW(args)
```


After define type model argument, we define argument model arguments.

```python
# Model Parmeters
parser.add_argument('--n_epochs', type=int, default=1000,
					help='max epochs')

parser.add_argument('--batch_size', type=int, default=32,
					help='each batch size')

parser.add_argument('--g_lr', type=float, default=0.0025,
					help='generator learning rate')

parser.add_argument('--e_lr', type=float, default=1e-4,
					help='encoder learning rate')

parser.add_argument('--d_lr', type=float, default=0.001,
					help='discriminator learning rate')

parser.add_argument('--beta', type=tuple, default=(0.5, 0.5),
					help='beta for adam')

parser.add_argument('--d_thresh', type=float, default=0.8,
					help='for balance dsicriminator and generator')

parser.add_argument('--z_size', type=float, default=200,
					help='latent space size')

parser.add_argument('--z_dis', type=str, default="norm", choices=["norm", "uni"],
					help='uniform: uni, normal: norm')

parser.add_argument('--bias', type=str2bool, default=False,
					help='using cnn bias')

parser.add_argument('--leak_value', type=float, default=0.2,
					help='leakeay relu')

parser.add_argument('--cube_len', type=float, default=32,
					help='cube length')

parser.add_argument('--image_size', type=float, default=224,
					help='cube length')

parser.add_argument('--obj', type=str, default="chair",
					help='tranining dataset object category')

parser.add_argument('--soft_label', type=str2bool, default=True,
					help='using soft_label')

parser.add_argument('--lrsh', type=str2bool, default=True,
					help='for learning rate shecduler')
```

Next, we must create file to save our output 3D object during training, logs for tensorboard and indicate training dataset.

```python
# dir parameters

parser.add_argument('--output_dir', type=str, default="../output",
					help='output path')

parser.add_argument('--input_dir', type=str, default='../input',
					help='input path')

parser.add_argument('--pickle_dir', type=str, default='/pickle/',
					help='input path')

parser.add_argument('--log_dir', type=str, default='/log/',
					help='for tensorboard log path save in output_dir + log_dir')

parser.add_argument('--image_dir', type=str, default='/image/',
					help='for output image path save in output_dir + image_dir')

parser.add_argument('--data_dir', type=str, default='/chair/',
					help='dataset load path')
```

We have another parameter about pickle step epoch for saving pickle for example which we can see in ``main.py``. For my training, i use this parameters:
``python main.py --alg_type 3DGAN --n_epochs 1000 --soft_label False --model_name 3DGAN --batch_size 32 --g_lr 0.0025 --e_lr 1e-4 --d_lr 0.001 --beta (0.5, 0.5) --d_thresh 0.8 --z_size 200``

# Results

![[result.png]]

The experiment aims to answer the question whether integration of automatic encoding architectures helps learn better parameters to generate new and realistic 3D objects from random latent vectors. An empirical evaluation of the figure above reveals some interesting aspects. During the training, each algorithm received a random latent vector $z_{t}$ at epochs 500, 1000 and 2000, and the generated 3D objects are shown in the figure. Although it may be difficult to concretely say which algorithm has the best 3D generated object, it is certain which models had the worst. In the in the early and later eras, the 3D-GAN algorithm was unable to produce an object that resembles a chair, meaning it took longer to practice.  The 3D-VAE-GAN in the 2000 era and the MV-3D-VAEGAN in the 500 era had the most realistic and original chairs. This may imply that MV-3DVAE-GAN can learn in fewer epochs than all other algorithms.

An important distinction between algorithms is the calculation time to train. There is a slight calculation increase between 3D-GAN and 3D-VAE-GAN due at the auto-encoding stage, but there is a massive increase in calculation time between the 3D-VAE-GAN algorithms and the two MV3D-VAE-GAN algorithms. This is due to the fact that each the view is coded separately and not vectorized. This leaves a direction of future work for multi-view algorithms. So, if computation time is a factor of interest, the 3DVAE-GAN model would be the algorithm of choice for this experience.

# Sources
[1]  Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik G. Learned-Miller. Multi-view convolutional neural networks for 3d shape recognition. In Proc. ICCV, 2015.
[2] Fakir S. Nooruddin and Greg Turk. Simplification and repair of polygonal models using volumetric techniques. 4 IEEE Transactions on Visualization and Computer Graphics, 9(2):191–205, 2003.
[3] Daniel Maturana. binvox-rw-py. https://github. com/dimatura/binvox-rw-py, 2016

https://github.com/black0017/3D-GAN-pytorch
https://github.com/rimchang/3DGAN-Pytorch/blob/master/3D_GAN/train.py