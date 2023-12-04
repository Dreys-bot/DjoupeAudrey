---
layout: '@/templates/BasePost.astro'
title: 3D Generative Adversial Model
description: This article presents an introduction to generative adversarial networks (GANs), a cutting-edge deep learning technique. GANs utilize an adversarial game between two neural networks to generate synthetic data. Applications include image, audio and video synthesis. The goal is to provide a high-level overview of GANs' generative process and vast potential.
pubDate: 2020-12-04T00:00:00Z
imgSrc: '/assets/images/image-post2.jpeg'
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