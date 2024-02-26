---
layout: '@/templates/BasePost.astro'
title: Neural style transfert
description: This article presents an introduction to generative adversarial networks (GANs), a cutting-edge deep learning technique. GANs utilize an adversarial game between two neural networks to generate synthetic data.
pubDate: 2023-12-04T00:00:00Z
imgSrc: '/assets/images/paint-icon.png'
imgAlt: 'Paint icon'
tags: ['python']
---

### 🎨🖌 Creating Art with the help of Artificial Intelligence !

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/ezgif.com-video-to-gif.gif]]

This repository contains an implementation of the Neural Style Transfer technique, a fascinating deep learning algorithm that combines the content of one image with the style of another image. By leveraging convolutional neural networks, this project enables you to create unique and visually appealing artworks that merge the content and style of different images.
<br> <!-- line break -->
![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/result.png]]

<br> <!-- line break -->


## 🎯 Objective 
The main goal of this project is to explore Neural-style-transfer through implementation. We'll Implement a NST model using Tensorflow and keras, and at the end of the project we'll deploy it as a web app so that anyone can create stunning digital art which they could even sell as NFT's.


## 📝 Summary of Neural Style Transfer

Style transfer is a computer vision technique that takes two images — a "content image" and "style image" — and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image. Training a style transfer model requires two networks,which follow a encoder-decoder architecture : 
- A pre-trained feature extractor 
- A transfer network


![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/nst%20architecture.jpg]]

<br> <!-- line break -->



***VGG19*** is used for Neural Style Transfert. It is a ***convolutional neural network*** that is trained on more than a million images from the ImageNet database. 

The network is 19 layers deep and trained on millions of images. Because of which it is able to detect high-level features in an image.  
Now, this ‘encoding nature’ of CNN’s is the key in Neural Style Transfer. Firstly, we initialize a noisy image, which is going to be our output image(G). We then calculate how similar is this image to the content and style image at a particular layer in the network(VGG network). Since we want that our output image(G) should have the content of the content image(C) and style of style image(S) we calculate the loss of generated image(G) w.r.t to the respective content(C) and style(S) image.  
Having the above intuition, let’s define our Content Loss and Style loss to randomly generated noisy image.

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/Pasted%20image%2020230823111307.png]]
<br> <!-- line break -->


### Content Loss

Calculating content loss means how similar is the randomly generated noisy image(G) to the content image(C).In order to calculate content loss:

Assume that we choose a hidden layer (L) in a pre-trained network(VGG network) to compute the loss.Therefore, let P and F be the original image and the image that is generated.And, F[l] and P[l] be feature representation of the respective images in layer L.Now,the content loss is defined as follows:

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/0_PJK8-P3tBWrUV1q1.png]]

### Style Loss

Before calculating style loss, let’s see what is the meaning of “**style of a image**” or how we capture style of an image.

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/0_dyVKNRn36XORjr9v.png]]


This image shows different channels or feature maps or filters at a particular chosen layer l.Now, in order to capture the style of an image we would calculate how “correlated” these filters are to each other meaning how similar are these feature maps.**But what is meant by correlation ?**

**Let’s understand it with the help of an example:**

Let the first two channel in the above image be Red and Yellow.Suppose, the red channel captures some simple feature (say, vertical lines) and if these two channels were correlated then whenever in the image there is a vertical lines that is detected by Red channel then there will be a Yellow-ish effect of the second channel.

Now,let’s look at how to calculate these correlations (mathematically).

In-order to calculate a correlation between different filters or channels we calculate the dot-product between the vectors of the activations of the two filters.The matrix thus obtained is called **Gram Matrix**.

**But how do we know whether they are correlated or not ?**

If the dot-product across the activation of two filters is large then two channels are said to be correlated and if it is small then the images are un-correlated.

**Putting it mathematically**

**Gram Matrix of Style Image(S):**

Here k and k’ represents different filters or channels of the layer L. Let’s call this Gkk’[l][S].

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/0_L8Y_zB0tWkcxFKMh.png]]
                    

**Gram Matrix for Generated Image(G):**

Here k and k’ represents different filters or channels of the layer L.Let’s call this Gkk’[l][G].

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/0_yjkYrNf7A_oMB_2V.png]]


Now,we are in the position to define Style loss:

Cost function between Style and Generated Image is the square of difference between the Gram Matrix of the style Image with the Gram Matrix of generated Image.

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/0_2LrpMFwbhD8OePdd.png]]

# Total Loss Function :

The total loss function is the sum of the cost of the content and the style image.Mathematically,it can be expressed as :

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/0_JPXny-rYTIeZRSb4.png]]


## To run locally

1. Download the pre-trained TF model.

    - The 'model' directory already contains the pre-trained model,but you can also download the pre-trained model from [here](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2).

2. Import this repository using git command
```
git clone https://github.com/deepeshdm/Neural-Style-Transfer.git
```
3. Install all the required dependencies inside a virtual environment
```
pip install -r requirements.txt
```
4. Copy the below code snippet and pass the required variable values
```python
import matplotlib.pylab as plt
from API import transfer_style

# Path of the downloaded pre-trained model or 'model' directory
model_path = r"C:\Users\Desktop\magenta_arbitrary-image-stylization-v1-256_2"

# NOTE : Works only for '.jpg' and '.png' extensions,other formats may give error
content_image_path = r"C:\Users\Pictures\my_pic.jpg"
style_image_path = r"C:\Users\Desktop\images\mona-lisa.jpg"

img = transfer_style(content_image_path,style_image_path,model_path)
# Saving the generated image
plt.imsave('stylized_image.jpeg',img)
plt.imshow(img)
plt.show()
```

## 🔥 Web Interface & API

In order to make it easy for anyone to interact with the model,we created a clean web interface using flask.

![[https://github.com/Dreys-bot/Neural-Style-Transfert/blob/main/ezgif.com-video-to-gif.gif]]















