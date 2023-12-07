---
layout: '@/templates/BasePost.astro'
title: 3D Generative Adversial Model
description: This article presents an introduction to generative adversarial networks (GANs), a cutting-edge deep learning technique. GANs utilize an adversarial game between two neural networks to generate synthetic data. Applications include image, audio and video synthesis. The goal is to provide a high-level overview of GANs' generative process and vast potential.
pubDate: 2023-12-04T00:00:00Z
imgSrc: '/assets/images/3D-gan-model.jpg'
imgAlt: 'Image post 2'
---

# What is GAN model?

![Global GAN Model](/assets/images/globalGANModel.png)
**GAN** is a machine learning model in which two **neural networks** compete with each other by using ***deep learning*** methods to become more accurate in their predictions. GANs typically run unsupervised and use a cooperative ***zero-sum game framework*** to learn, where one person's gain equals another person's loss.

GANs consist of two models, namely, the **generative model** and the **discriminator model**. On the one hand, the generative model is responsible for creating fake data instances that resemble your training data. On the other hand, the discriminator model behaves as a classifier that distinguishes between real data instances from the output of the generator. The generator attempts to deceive the discriminator by generating real images as far as possible, and the discriminator tries to keep from being deceived.

The discriminator penalizes the generator for producing an absurd output. At the initial stages of the training process, the generator generates fake data, and the discriminator quickly learns to tell that it’s fake. But as the training progresses, the generator moves closer to producing an output that can fool the discriminator. Finally, if generator training goes well, then the discriminator performance gets worse because it can’t quickly tell the difference between real and fake. It starts to classify the fake data as real, and its accuracy decreases. Below is a picture of the whole system:

![GAN process](/assets/images/GANProcess.png)

# How does GAN Model works?

Building block of GAN are composed with 2 neural networks working together.

**1. Generator:** Model that learns to make fake things to look real

**2. Discriminator:** Model that learns to differentiate real from fake

>***The goal of generator is to fool the discriminator while discriminator's goal is to distinguish betwen real from fake***

The keep compete between each other until at the end fakes (generator by generator) look real (discriminator can't differentiate).

![GAN diagram](/assets/images/GANDiagram.png)
**We notice that what we input to generator is **Noise**, why?**
**Noise** in this scenario, we can think about it as random small number vector. When we vary the noise on each run(training), it helps ensure that generator will generate different image on the same class on the same class based on the data that feed into discriminator and got feed back to the generator.

![noise generator](/assets/images/noiseGenerator.png)

Then, generate will likely generate the object that are common to find features in the dataset. For example, 2 ears with round eye of cat rather with common color rather than sphinx cat image that might pretty be rare in the dataset.
![ganNetwork](/assets/images/ganNetwork.png)

The generator model generated images from **random noise(z)** and then learns how to generate realistic images. Random noise which is input is sampled using uniform or normal distribution an dthen it is fed into the generator which generated an image. The generator output which are fake images and the real images from the training set is fed into the discriminator that learns how to differentiate fake images from real images. The output **D(x)** is the probability that the input is real. If the input is real, **D(x)** would be 1 and if it is generated, **D(x)** should be 0.

# Types of GAN models

## Deep Convolutional Generative Adversial Network
DCGAN stands for Deep Convolutional Generative Adversarial Network. It is a type of GAN that uses convolutional layers in both the generative and discriminative models.

In a DCGAN, the generative model, G, is a deep convolutional neural network that takes as input a random noise vector, z, and outputs a synthetic image. The goal of G is to produce synthetic images that are similar to the real images in the training data.
The discriminative model, D, is also a deep convolutional neural network that takes as input an image, either real or synthetic, and outputs a probability that the image is real. The goal of D is to correctly classify real images as real and synthetic images as fake.

The overall loss function for a DCGAN is defined as the sum of the loss functions for G and D. The loss function for G is defined as:

$L_G = E[log{(1 - D(G(z)))}]$


This loss function encourages G to produce synthetic images that are classified as real by D. In other words, it encourages G to generate images that are similar to the real images in the training data.

The loss function for D is defined as:

$L_G = E[log(D(x))] + E[log(1 - D(G(z)))]$

This loss function encourages D to correctly classify real images as real and synthetic images as fake. In other words, it encourages D to accurately differentiate between real and fake images.

The overall loss function for the DCGAN is then defined as:

$L_{DCGAN} = L_{G} + L_{D}$

This loss function is minimized during training by updating the weights of G and D using gradient descent. By minimizing this loss function, the DCGAN learns to generate high-quality synthetic images that are similar to the real images in the training data.

## CapsGAN

Deep learning techniques like CNN have outperformed classical machine learning techniques for the image classification task. However, a new network called [CapsNet](https://www.cs.toronto.edu/~hinton/absps/transauto6.pdf) has outperformed CNNs for this task. The limitation with a CNN for the object recognition task is that the activation of its neurons is based on the chances of detecting specific image features. Neurons do not consider the properties or features of an image, such as pose, texture, and deformation of the objects in the image. 
In other words, CNNs are incapable because of their invariance as a result of the pooling operation. Basically, a capsule is a group of neural layers. While a typical neuron outputs a single scalar value, a capsule outputs a vector representing a generalized set of related object properties. A capsule attempts to capture many object properties like pose (position, angle of view, size), deformation, the texture inside an image to define the probability of some object existence. Transforming autoencoders (_refer AEGAN section for a brief introduction to autoencoders_) makes use of this complex feature detectors called a capsule, to explicitly capture the exact pose of each feature in the image and this is how they try to learn the overall transformation matrix. 
Capsules allow the autoencoder to maintain translational invariance without throwing away important positional information. They are not only capable of recognizing features in different poses and lighting conditions but are also capable of outputting pose-specific variables to be used by higher visual layers rather than discarding them. The goal was not to recognize objects in images but to accept an image and its pose as input and output the same image in the original pose.

## AEGAN

Autoencoding Generative Adversarial Networks (AEGAN) is a four-network model comprising of two GANs and two autoencoders as shown below:

![AGAN](/assets/images/AEGAN.png)

Just like GANs, autoencoders are a type of unsupervised learning algorithms. The autoencoders consist of two virtual components in its network, namely, the encoder model and the decoder model. The encoder model maps the input data to the network’s internal representation, just like the notion of data compression operation, and the decoder model tries to reconstruct the input from the network’s internal data representation just like the notion of data decompression operation. Therefore, the output shape of the autoencoder is the same as the input, that allows the network to learn basic representations better.
AEGAN leverages the advantages of GANs and autoencoders by stabilizing the GAN training and thereby overcomes the common problems of GANs, namely, mode collapse and lack of convergence.