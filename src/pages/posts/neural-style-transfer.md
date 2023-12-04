---
layout: '@/templates/BasePost.astro'
title: Neural style transfer
description: This article explains neural style transfer, an AI technique combining the visual content of one image with the artistic style of another. It details how convolutional neural networks capture content and style, and how iterative optimization blends the two into a new hybrid image. A clear guide to this generative deep learning approach.
pubDate: 2023-09-03
imgSrc: '/assets/images/neural-style-transfer.jpg'
imgAlt: 'Image post 4'
---

# Introduction

![Style Neural Network results](/assets/images/styleNeuralNetResult.png)

As we can see, the generated image is having the content of the ***Content image and style image***. This above result cannot be obtained by overlapping the image. So the main questions are:  ***how we make sure that the generated image has the content and style of the image?  how we capture the content and style of respective images?***

# What Convolutional Neural Network Capture?

![CNN Architecture](/assets/images/CNN_architecture.png)

Convolutional neural networks progressively learn to represent image features. At level 1, with 32 filters, the network can capture simple patterns such as straight or horizontal lines. While these elements may not seem relevant to the human eye, they are essential for the network's learning.

At level 2, with 64 filters, the network starts to perceive more complex features, such as a dog's head or a car wheel. This increasing ability to extract simple and complex elements constitutes what is called "feature representation".

It is important to note that CNNs do not aim to identify images, but rather learn to encode what they represent. This encoding capacity underlies their application to neural style transfer.

Let us explore more deeply this fundamental concept of CNNs. Through their progressive approach, they build an ever richer representation of images as layers deepen, underpinning their power for tasks such as classification or style transfer.


# How Convolutional Neural Networks are used to capture content and style of images?

***VGG19*** is used for Neural Style Transfert. It is a ***convolutional neural network*** that is trained on more than a million images from the ImageNet database. 

Now, this "encoding nature" of CNNs is key for Neural Style Transfer. First, we initialize a noisy image that will be our output image (G). We then calculate how similar this image is to the content and style images at particular layers in the network (the VGG network). Since we want our output image (G) to have the content of the content image (C) and the style of the style image (S), we calculate the loss of the generated image (G) with respect to the respective content (C) and style (S) images.

With this intuition in mind, let's define our content loss and style loss for the randomly generated noisy image.

![VGG](/assets/images/VGG.png)

## Content Loss

Calculating content loss means how similar is the randomly generated noisy image(G) to the content image(C).In order to calculate content loss :

Assume that we choose a hidden layer (L) in a pre-trained network(VGG network) to compute the loss.Therefore, let P and F be the original image and the image that is generated, and, F[l] and P[l] be feature representation of the respective images in layer L. Now, the content loss is :

$L_{content}(\vec{p},\vec{x},l) = \frac{1}{2}\sum_{i,j}(F_{ij}^{l} - P_{ij}^{l})^2$

## Style Loss

Before calculating style loss, let’s see what is the meaning of “**style of a image**” or how we capture style of an image.

### How we capture style of an image ?

![Different channels or Feature maps in layer l](https://miro.medium.com/v2/resize:fit:264/0*dyVKNRn36XORjr9v.png)



This image shows different channels or feature maps or filters at a particular chosen layer **l**. Now, in order to capture the style of an image we would calculate how **correlated** these filters are to each other meaning how similar are these feature maps. **But what is meant by correlation ?**

Let’s understand it with the help of an example:

Let the first two channel in the above image be Red and Yellow. Suppose, the red channel captures some simple feature (say, vertical lines) and if these two channels were correlated then whenever in the image there is a vertical lines that is detected by Red channel then there will be a Yellow-ish effect of the second channel.

Now,let’s look at how to calculate these correlations (mathematically).

In-order to calculate a correlation between different filters or channels, we calculate the dot-product between the vectors of the activations of the two filters.The matrix thus obtained is called **Gram Matrix**.

**But how do we know whether they are correlated or not ?**

If the dot-product across the activation of two filters is large then two channels are said to be correlated and if it is small then the images are un-correlated. **Putting it mathematically :**

**Gram Matrix of Style Image(S)**:

Here k and k’ represents different filters or channels of the layer L. Let’s call this Gkk’[l][S].

$J_{style}=\sum_{l}^H\sum_{j}^W(A_{ijk}^{[l][S]} - A_{ijk'}^{[l][S]})$


**Gram Matrix for Generated Image(G)**:

Here k and k’ represents different filters or channels of the layer L.Let’s call this Gkk’[l][G].

$J_{style.generated}=\sum_{l}^H\sum_{j}^W(A_{ijk}^{[l][G]} - A_{ijk'}^{[l][G]})$

Now,we are in the position to define Style loss:

Cost function between Style and Generated Image is the square of difference between the Gram Matrix of the style Image with the Gram Matrix of generated Image.

$J_{S,G} = \frac{1}{2H^lW^lC^l}\sum_{k}\sum_{k'}(G_{kk'}^{[l][S]} - G_{kk'}^{[l][G]})$

Now,Let’s define the total loss for Neural Style Transfer.

## Total Loss Function :

The total loss function is the sum of the cost of the content and the style image.Mathematically,it can be expressed as :

$L_{total}(\vec{p}, \vec{α},\vec{x}) = αL_{content}(\vec{p}, \vec{x}) + βL_{style}(\vec{α}, \vec{x})$

You may have noticed Alpha and beta in the above equation.They are used for weighing Content and Style cost respectively.In general,they define the weightage of each cost in the Generated output image.

Once the loss is calculated,then this loss can be minimized using **backpropagation** which in turn will optimize our **randomly generated image** into a **meaningful piece of art**.

This sums up the working of Neural Style Transfer.


# Conclusion

In conclusion, this article has explored the fundamental principle behind neural style transfer and its ability to successfully combine the content of one image with the style of another.
We began by understanding how convolutional neural networks progressively learn to represent image features through the extraction of simple to complex patterns. This encoding capability is crucial for separating content and style in an unsupervised manner.
Precise definitions of content and style losses were provided, mathematically modeling the desired similarity between the generated image and the reference images. The calculation of the Gram matrix emerged as central to quantifying correlations between filters indicative of style.
Optimizing the generated image through backpropagation of the overall cost function gradient then allows synthesizing a new image realistically fusing the targeted characteristics.

While still improvable, this unsupervised approach opens promising avenues for numerous artistic and industrial applications. Future work could focus on enhancing its capacities.
Through this article, we aimed to provide a clear and detailed explanation of the internal workings behind neural style transfer, foundational to many deep learning innovations. Continued research in this area holds potential for generating increasingly convincing and customized artistic blends.