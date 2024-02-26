---
layout: '@/templates/BasePost.astro'
title: Neural style transfer
description: This article explains neural style transfer, an AI technique combining the visual content of one image with the artistic style of another. It details how convolutional neural networks capture content and style, and how iterative optimization blends the two into a new hybrid image. A clear guide to this generative deep learning approach.
pubDate: 2023-09-03
imgSrc: '/assets/images/neural-style-transfer/index.jpg'
imgAlt: "Vladimir Putin's photo modified by artificial intelligence with an aquarel style"
---

# Introduction
To generally separate content from style in natural images is still an extremely difficult problem. However, the recent advance of Deep Convolutional Neural Networks has produced powerful computer vision systems that learn to extract high-level semantic information from natural images.
Transferring the style from one image onto another can be considered a problem of texture transfer. In texture transfer the goal is to synthesise a texture from a source image while constraining the texture synthesis in order to preserve the semantic content of a target image.
For texture synthesis there exist a large range of powerful non-parametric algorithms that can synthesise photorealistic natural textures by resampling the pixels of a given source texture. Therefore, a fundamental prerequisite is to find image representations that independently model variations in the semantic image content and the style in which is presented.

![Style Neural Network results](/assets/images/neural-style-transfer/styleNeuralNetResult.png)

As we can see, the generated image is having the content of the ***Content image and style image***. This above result cannot be obtained by overlapping the image. So the main questions are:  ***how we make sure that the generated image has the content and style of the image?  how we capture the content and style of respective images?***

# What Convolutional Neural Network Capture?

![CNN Architecture](/assets/images/neural-style-transfer/CNN_architecture.png)

Convolutional neural networks progressively learn to represent image features. At level 1, with 32 filters, the network can capture simple patterns such as straight or horizontal lines. While these elements may not seem relevant to the human eye, they are essential for the network's learning.

At level 2, with 64 filters, the network starts to perceive more complex features, such as a dog's head or a car wheel. This increasing ability to extract simple and complex elements constitutes what is called "feature representation".

It is important to note that CNNs do not aim to identify images, but rather learn to encode what they represent. This encoding capacity underlies their application to neural style transfer.

Let us explore more deeply this fundamental concept of CNNs. Through their progressive approach, they build an ever richer representation of images as layers deepen, underpinning their power for tasks such as classification or style transfer.


# How Convolutional Neural Networks are used to capture content and style of images?

![CNN_VGG](/assets/images/neural-style-transfer/CNN_VGG.png)

The figure showed an image representations in a Convolutional Neural Network (CNN). A given input image is represented as a set of filtered images at each processing stage in the CNN. While the number of different filters increases along the processing hierarchy, the size of the filtered images is reduced by some downsampling mechanism (e.g. max-pooling) leading to a decrease in the total number of units per layer of the network.

**Content Reconstructions**. We can visualise the information at different processing stages in the CNN by reconstructing the input image from only knowing the network’s responses in a particular layer. We reconstruct the input image from from layers ‘conv1 2’ (a), ‘conv2 2’ (b), ‘conv3 2’ (c), ‘conv4 2’ (d) and ‘conv5 2’ (e) of the original VGG-Network. We find that reconstruction from lower layers is almost perfect (a–c). In higher layers of the network, detailed pixel information is lost while the high-level content of the image is preserved (d,e).

**Style Reconstructions**. On top of the original CNN activations we use a feature space that captures the texture information of an input image. The style representation computes correlations between the different features in different layers of the CNN. We reconstruct the style of the input image from a style representation built on different subsets of CNN layers ( ‘conv1 1’ (a), ‘conv1 1’ and ‘conv2 1’ (b), ‘conv1 1’, ‘conv2 1’ and ‘conv3 1’ (c), ‘conv1 1’, ‘conv2 1’, ‘conv3 1’ and ‘conv4 1’ (d), ‘conv1 1’, ‘conv2 1’, ‘conv3 1’, ‘conv4 1’ and ‘conv5 1’ (e). This creates images that match the style of a given image on an increasing scale while discarding information of the global arrangement of the scene

## Content Loss

Generally each layer in the network defines a non-linear filter bank whose complexity increases with the position of the layer in the network. Hence a given input image $\vec{x}$ is encoded in each layer of the Convolutional Neural Network by the filter responses to that image. A layer with $N_l$ distinct filters has $N_l$ feature maps each of size Ml , where $M_l$ is the height times the width of the feature map. So the responses in a layer $l$ can be stored in a matrix $F_l$ where $F^l_{ij}$ is the activation of the $i^{th}$ filter at position $j$ in layer $l$.

To visualise the image information that is encoded at different layers of the hierarchy one can perform gradient descent on a white noise image to find another image that matches the feature responses of the original image.
Let  $\vec{p}$ and  $\vec{x}$ be the original image and the image that is generated, and $P_l$ and $F_l$ their respective feature representation in layer $l$. We then define the squared-error loss between the two feature representations.


$L_{content}(\vec{p},\vec{x},l) = \frac{1}{2}\sum_{i,j}(F_{ij}^{l} - P_{ij}^{l})^2$

The derivative of this loss with respect to the activations in layer $l$ equals

$\frac{\partial \mathcal{L}_{\text {content }}}{\partial F_{i j}^l}= \begin{cases}\left(F^l-P^l\right)_{i j} & \text { if } F_{i j}^l>0 \\ 0 & \text { if } F_{i j}^l<0,\end{cases}$

from which the gradient with respect to the image  $\vec{x}$ can be computed using standard error back-propagation  

Thus we can change the initially random image  $\vec{x}$ until it generates the same response in a certain layer of the Convolutional Neural Network as the original image  $\vec{p}$. When Convolutional Neural Networks are trained on object recognition, they develop a representation of the image that makes object information increasingly explicit along the processing hierarchy. Therefore, along the processing hierarchy of the network, the input image is transformed into representations that are increasingly sensitive to the actual content of the image, but become relatively invariant to its precise appearance. Thus, higher layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction very much. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image. We therefore refer to the feature responses in higher layers of the network as the ***content representation***

## Style Loss

To obtain a representation of the style of an input image, we use a feature space designed to capture texture information . This feature space can be built on top of the filter responses in any layer of the network. It consists of the correlations between the different filter responses, where the expectation is taken over the spatial extent of the feature maps. These feature correlations are given by the Gram matrix.

By including the feature correlations of multiple layers, we obtain a stationary, multi-scale representation of the input image, which captures its texture information but not the global arrangement. Again, we can visualise the information captured by these style feature spaces built on different layers of the network by constructing an image that matches the style representation of a given input image (Fig 1, style reconstructions). This is done by using gradient descent from a white noise image to minimise the mean-squared distance between the entries of the Gram matrices from the original image and the Gram matrices of the image to be generated. 

Before calculating style loss, let’s see what is the meaning of “**style of a image**” or how we capture style of an image.

### How we capture style of an image ?

![Different channels or Feature maps in layer l](https://miro.medium.com/v2/resize:fit:264/0*dyVKNRn36XORjr9v.png)



This image shows different channels or feature maps or filters at a particular chosen layer **l**. Now, in order to capture the style of an image we would calculate how **correlated** these filters are to each other meaning how similar are these feature maps. **But what is meant by correlation?**

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

$L_{total}(\vec{p}, \vec{\alpha},\vec{x}) = \alpha L_{content}(\vec{p}, \vec{x}) + \beta L_{style}(\vec{\alpha}, \vec{x})$

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