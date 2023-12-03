---
layout: '@/templates/BasePost.astro'
title: 3D GAN Model
description: This article explains neural style transfer, an AI technique combining the visual content of one image with the artistic style of another. It details how convolutional neural networks capture content and style, and how iterative optimization blends the two into a new hybrid image. A clear guide to this generative deep learning approach.
pubDate: 2023-09-01
imgSrc: '/assets/images/project-web-design.png'
imgAlt: 'Project Web Design'
tags: 
  - 3D
  - GAN
  - SkitLearn
  - NumPy
  - Matplotlib
  - SkyPy
  - Pytorch
---

# Summary
---
🚩[[#Introduction to 3D-GAN basic knowledge]]
		☑️[[#3D Convolution]]
		☑️[[#3D-GAN architecture]]
		☑️[[#The architecture of generator network]]
		☑️[[#The architecture of the discriminator network]]
		☑️[[#Objective function]]
		☑️[[# Steps Training 3D-GAN]]
📽️[[#Create project]]
📄[[#Prepare data]]
🤖[[#Keras implementation of 3D-GAN]]
⚙️[[#Training 3D-GAN]]
😊[[#Hyperparameter optimization]]
🤔[[#Pratical applications of 3D-GAN]]

3D-GAN is a GAN architecture for 3D shape generation. 3D shape generation is generally a complex problem due to the complexity involved in processing 3D images. 3D-GAN is a solution that can generate realistic, changing 3D shapes. we will implement 3D-GAN using the Keras framework.

# Introduction

**3D Generative Adversarial Network** ( **3D-GAN** ) is a variant of GAN, just like StackGAN, CycleGAN and **Super-Resolution Generative Adversarial Network** ( **SRGAN** ). Similar to Naive GAN, it has generator and discriminator models. Both networks use 3D convolutional layers instead of 2D convolutions. If given enough data, it can learn to generate 3D shapes with good visual quality.

Before taking a closer look at the 3D-GAN network, let us understand 3D convolution.

## 3D Convolution

In short, the 3D convolution operation applies 3D filters to the input data along these three `x`directions `y`. `z`This operation creates a stacked list of 3D feature maps. The output shape is similar to that of a cube or cuboid. The figure below illustrates the 3D convolution operation. The highlighted portion of the left cube is the input data. The kernel is in the middle and has the shape of `(3, 3, 3)`. The block on the right is the output of the convolution operation:

![[Pasted image 20231120172253.png]]

Now that we have a basic understanding of 3D convolution, let's move on to look at the architecture of 3D-GAN.

## 3D-GAN architecture

Both networks in 3D-GAN are deep convolutional neural networks. The generator network is usually an upsampling network. It upsamples noise vectors (vectors from a probabilistic latent space) to generate a 3D image of shape , whose length, width, height, and channels are similar to the input image. The discriminator network is a downsampling network. Using a series of 3D convolution operations and dense layers, it can identify whether the input data provided to it is real or fake.

In the next two sections, we introduce the architecture of the generator and discriminator networks.

## The architecture of generator network

The generator network consists of five volumetric fully convolutional layers with the following configuration:

- **Convolutional layers** : 5
- **Filters** : 512, 256, 128 and 64, 1
- **Kernel size** : `4 x 4 x 4`, , , , `4 x 4 x 4` `4 x 4 x 4` `4 x 4 x 4` `4 x 4 x 4`
- **Stride** : 1, 2, 2, 2, 2 or`(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)`
- **Batch normalization** : yes, yes, yes, yes, no
- **Activation** : ReLU, ReLU, ReLU, ReLU, Sigmoid
- **Pooling layer** : no, no, no, no, no
- **Linear layer** : no, no, no, no, no

The input and output of the network are as follows:

- **Input** : 200-dimensional vector sampled from the probabilistic latent space
- **Output** : `64x64x64`3D image of shape

The following diagram shows the architecture of the generator:

![[Pasted image 20231120173703.png]]

The figure below shows the flow of tensors in the generator network and the input and output shapes of the tensors for each layer. This will give you a better understanding of the network:

![[Pasted image 20231120173908.png]]

## The architecture of the discriminator network

The discriminator network consists of five volumetric convolutional layers with the following configuration:

- **3D convolutional layers** : 5
- **Channel** : 64, 128, 256, 512, 1
- **Core size** : 4, 4, 4, 4, 4, 4
- **Stride** : 2, 2, 2, 2, 1
- **Activation** : LReLU, LReLU, LReLU, LReLU, Sigmoid
- **Batch Normalization** : Yes, Yes, Yes, Yes, No
- **Pooling layer** : no, no, no, no, no
- **Linear layer** : no, no, no, no, no

The input and output of the network are as follows:

- **Input** : `(64, 64, 64)`3D image of shape
- **Output** : The probability that the input data belongs to the real or fake class

The image belows shows tensorflow for each layer in the discriminator network and the input and output shapes of the tensors.

> [!info]
> The discriminator network mainly mirrors the generator network. An important difference is that it uses **LeakyReLU** instead of **RELU** as the activation function. Moreover, the **Sigmoid layer** at the end of the networks is used for binary classification and predicts whether the provides images is real or fake. The last layer has no normalization layer, but other layers use batch normalized input.

## Objective function

The objective function is the main method of training 3D-GAN. It provides loss values, which are used to calculate gradients and then update weight values. The adversarial loss function of 3D-GAN is as follows:

![[Pasted image 20231121153427.png]]

Here, is the binary cross-entropy loss or classification loss, is the adversarial loss, is the latent vector from the probability space , is the output of the discriminator network, and is the output of the generator network.

## Steps Training 3D-GAN

Training a 3D-GAN is similar to training a naive GAN. The steps involved in training 3D-GAN are as follows:

1. Sample a 200-dimensional noise vector from a Gaussian (normal) distribution.
2. Use a generator model to generate fake images.
3. Train the generator network on real images (sampled from real data) and fake images generated by the generator network.
4. Train a generator model using adversarial models. Don't train the discriminator model.
5. Repeat these steps for the specified number of cycles.

We'll explore these steps in detail in later sections. Let's go ahead and build a project.


# Create Projet

1. Clone the project repository at this link
```python
git clone 
```

2. Navigate to the parent directory as shown below:
```python
cd Generative-Adversarial-Networks-Projects
```

3. Change the directory from the currrent directory to chapter02
```python
cd chapter02
```

4. Create a python virtual environment for the project:
```python
conda create GAN_env
```

5. Activate conda environment
```python
conda activate GAN_env
```

6.Install requirements.txt

```python
pip install -r requirements.txt
```



# Prepare data

In this project, i use **3D ShapeNets** dataset, which is availables from this page[1]. It contains annotated 3D shapes for 40 object categories. In the next section, i will download, extract and explore the dataset. 

## Download and extract the dataset

1. First download using the link below **3DShapeNets**:
```python
wget http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip
```

2. Run the following command to extract the file into the appropirate directory:
```python
unzip 3DShapeNetsCode.zip
```

Now, we have successfully downloaded and extracted the dataset.It contains ``.mat`` images in (MATLAB) format.Every other image is a 3D image. In the next section, we will learn about voxels, which are points in 3D space.

## Explore datasets

To understand the dataset, we need to visualize the 3D images. In the next few sections, we will first look at what a voxel is in more detail. We will then load and visualize the 3D images.

## What is voxel?

A **voxel** is a point in three dimensional space. A voxel defines a position with three coordinates in the ``x``, ``y`` and directions ``z``. Voxels are the basic units for representing 3D images. They are mainly used in CAT scans, X-rays and MRIs to create accurate 3D models of the human boddy and other 3D objects. To process 3D images, it is important to understand voxels, as these are the components of a 3D images. The following image is included to give you an idea of what voxels in a 3D image look like:

![[Pasted image 20231123134442.png]]

A sequence of voxels in a 3D image. The shaded areas are individual voxels. The previous image is a stacked representation of voxels. The gray rectangle represents a voxel. Now that you understand what voxels are, let's load and visualize 3D images in the next section. 

## Load and visualize 3D images

The 3D shapeNets dataset contains ``.mat`` file formats.  We convert these .mat files into Numpy N-dimensional arrays. We will also visualize 3D images to gain an untuitive understanding of the dataset.

Let's execute this code to ``.mat`` in order to load a 3D image from a file:
1. Retrieve the function ``scipy`` in use . code show as below: ``loadmat() voxels``
```python
import scipy.io as io
voxels = io.loadmat("path to .mat file")['instance']
```

2. The loaded 3D image has a shape of `30x30x30`. Our network requires `64x64x64`images of shape . We will use NumPy's methods to increase the size of the 3D image : `pad()``32x32x32`
```python
import numpy as np
voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
```
`pad()`The method takes four parameters, which are an N-dimensional array of actual voxels, the number of values ​​that need to be filled to the edge of each axis, the mode value ( `constant`) and `constant_values` be filled.

3. Then, use the functions `scipy.ndimage`in the module `zoom()`to convert the 3D image into a `64x64x64`3D image of size.

```python
import scipy.ndimage as nd
voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
```
Our network requires images to be of shape `64x64x64`, which is why we convert 3D images into this shape.

## Visualize 3D images

Lets us visualize a 3D image using matplotlib as shown in the code:

1. First create a matplotlib figure and add subplot to it:
```python
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
```

2. Next, add voxels to the plot:
```python
ax.voxels(voxels, edgecolor="red")
```

3. Next, display the plot and save it as an images so we can visualize and understand it:
```python
plt.show() plt.savefig(file_path)
```

**Result**

![[Pasted image 20231123145646.png]]
![[Pasted image 20231123150120.png]]
![[Pasted image 20231123150236.png]]

We have successfully downloaded, extracted and browsed the dataset. We also looked at how to use voxels. In the next section, we will implement 3D-GAN in the Keras framework.

# Keras Implementation of 3D-GAN

In this section, we will implement the **generator network** and the **discriminator network** in the keras framework. We need to create two Keras framework according to the structure developped befor.  Both networks have independent weight values. Let's start with the generator network. 

## Generator network

To implement the generator network, we need to create a keras model and add neural network layers. The steps required to implement a generator network are as follows:

1. Start by specifying values for the different hyperparameters:
In the section before, we have seen the structure of ***generator network***. Now, we will implement it.

```python
z_size = 200 gen_filters = [512, 256, 128, 64, 1]
gen_kernel_sizes = [4, 4, 4, 4, 4]
gen_strides = [1, 2, 2, 2, 2]
gen_input_shape = (1, 1, 1, z_size)
gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
gen_convolutional_blocks = 5
```

2. Next, create an input layer to allow input to the network. The inputs to the generator network are vectors sampled from the probabilistic latent space:
```python
input_layer = Input(shape=gen_input_shape)
```

3. Then, add the first 3D transposed convolution block, as shown in the following code:
```python
## First 3D transpose convolution( or 3D deconvolution) block a = Deconv3D(filters=gen_filters[0],  
 kernel_size=gen_kernel_sizes[0],
             strides=gen_strides[0])(input_layer)
a = BatchNormalization()(a, training=True)
a = Activation(activation=gen_activations[0])(a)
```

4. Next, add four more 3D transpose convolution blocks as follows:
```python
## Next 4 3D transpose convolution( or 3D deconvolution) blocks for i in range(gen_convolutional_blocks - 1):
    a = Deconv3D(filters=gen_filters[i + 1], 
 kernel_size=gen_kernel_sizes[i + 1],
                 strides=gen_strides[i + 1], padding='same')(a)
    a = BatchNormalization()(a, training=True)
    a = Activation(activation=gen_activations[i + 1])(a)
```

5. Then, create Keras model and specify the inputs and ouputs of the generator network:

```python
model = Model(inputs=input_layer, outputs=a)
```

6. Wrap the entire code of the generator network inside a ``build_generator`` function called: 

```python
def build_generator():
 """
 Create a Generator Model with hyperparameters values defined as follows  :return: Generator network
 """  z_size = 200
  gen_filters = [512, 256, 128, 64, 1]
 gen_kernel_sizes = [4, 4, 4, 4, 4]
 gen_strides = [1, 2, 2, 2, 2]
 gen_input_shape = (1, 1, 1, z_size)
 gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
 gen_convolutional_blocks = 5    input_layer = Input(shape=gen_input_shape)
 
## First 3D transpose convolution(or 3D deconvolution) block
  a = Deconv3D(filters=gen_filters[0], 
 kernel_size=gen_kernel_sizes[0],
 strides=gen_strides[0])(input_layer)
 a = BatchNormalization()(a, training=True)
 a = Activation(activation='relu')(a)
 
## Next 4 3D transpose convolution(or 3D deconvolution) blocks
  for i in range(gen_convolutional_blocks - 1):
 a = Deconv3D(filters=gen_filters[i + 1], 
 kernel_size=gen_kernel_sizes[i + 1],
 strides=gen_strides[i + 1], padding='same')(a)
 a = BatchNormalization()(a, training=True)
 a = Activation(activation=gen_activations[i + 1])(a)
 gen_model = Model(inputs=input_layer, outputs=a)
 gen_model.summary()
 return gen_model
```

## Discriminator network

Liewise, to implement the discriminator network, we need to create a keras model and add neural networks layers to it. The steps required to implement the discriminator network are as follows:

1. Start by specifying values for the different hyperparameters:
```python
dis_input_shape = (64, 64, 64, 1)
dis_filters = [64, 128, 256, 512, 1]
dis_kernel_sizes = [4, 4, 4, 4, 4]
dis_strides = [2, 2, 2, 2, 1]
dis_paddings = ['same', 'same', 'same', 'same', 'valid']
dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid']
dis_convolutional_blocks = 5
```

2. Next, create an input layer to allow input to the network. The input to the discriminator network is a 3D image of shape ``64x64x64x1``
```python
dis_input_layer = Input(shape=dis_input_shape)
```

3. Then, add the first 3D convolution block as follows:

```python
## The first 3D Convolution block a = Conv3D(filters=dis_filters[0],
           kernel_size=dis_kernel_sizes[0],
           strides=dis_strides[0],
           padding=dis_paddings[0])(dis_input_layer)
a = BatchNormalization()(a, training=True)
a = LeakyReLU(alphas[0])(a)
```

4. After that, add four more 3D convolution blocks as follows:
```python
## The next 4 3D Convolutional Blocks for i in range(dis_convolutional_blocks - 1):
    a = Conv3D(filters=dis_filters[i + 1],
               kernel_size=dis_kernel_sizes[i + 1],
               strides=dis_strides[i + 1],
               padding=dis_paddings[i + 1])(a)
    a = BatchNormalization()(a, training=True)
    if dis_activations[i + 1] == 'leaky_relu':
        a = LeakyReLU(dis_alphas[i + 1])(a)
    elif dis_activations[i + 1] == 'sigmoid':
        a = Activation(activation='sigmoid')(a)
```

5. Next, create a keras model and specify the inputs and outputs for the discriminator network:
```python
dis_model = Model(inputs=dis_input_layer, outputs=a)
```

6. Wrap the complete code of the discriminator network in a function like this:
```python
def build_discriminator():
    """
 Create a Discriminator Model using hyperparameters values defined as follows  :return: Discriminator network
 """    dis_input_shape = (64, 64, 64, 1)
    dis_filters = [64, 128, 256, 512, 1]
    dis_kernel_sizes = [4, 4, 4, 4, 4]
    dis_strides = [2, 2, 2, 2, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 
 'leaky_relu', 'sigmoid']
    dis_convolutional_blocks = 5    dis_input_layer = Input(shape=dis_input_shape)
 
## The first 3D Convolutional block
  a = Conv3D(filters=dis_filters[0],
               kernel_size=dis_kernel_sizes[0],
               strides=dis_strides[0],
               padding=dis_paddings[0])(dis_input_layer)
    a = BatchNormalization()(a, training=True)
    a = LeakyReLU(dis_alphas[0])(a)
 
## Next 4 3D Convolutional Blocks
  for i in range(dis_convolutional_blocks - 1):
        a = Conv3D(filters=dis_filters[i + 1],
                   kernel_size=dis_kernel_sizes[i + 1],
                   strides=dis_strides[i + 1],
                   padding=dis_paddings[i + 1])(a)
        a = BatchNormalization()(a, training=True)
        if dis_activations[i + 1] == 'leaky_relu':
            a = LeakyReLU(dis_alphas[i + 1])(a)
        elif dis_activations[i + 1] == 'sigmoid':
            a = Activation(activation='sigmoid')(a)
    dis_model = Model(inputs=dis_input_layer, outputs=a)
    print(dis_model.summary())
 return dis_model
```

Now, we are reading to train 3D-GAN.

# Training 3D-GAN

Training a 3D-GAN is similar to training a naive GAN. We first train the discriminator network on generated and real images, but freeze the generator network. We then train the generator network but freeze the discriminator network. We repeat this process for the specified number of periods. In one iteration, we train two networks sequentially. Training 3D-GAN is an end-to-end training process. Let's go through these steps one by one.

## Train the network

To train 3D-GAN, follow these steps:

1. Start by specifying the values of the different hyperparameters required for training as follows: 
```python
gen_learning_rate = 0.0025 
dis_learning_rate = 0.00001
beta = 0.5
batch_size = 32
z_size = 200
DIR_PATH = 'Path to the 3DShapenets dataset directory'
generated_volumes_dir = 'generated_volumes'
log_dir = 'logs'
```

2. Next, create and compile two networks as follows:
```python
## Create instances
generator = build_generator()
discriminator = build_discriminator()
 
## Specify optimizer 
gen_optimizer = Adam(lr=gen_learning_rate, beta_1=beta)
dis_optimizer = Adam(lr=dis_learning_rate, beta_1=0.9)
 
## Compile networks
generator.compile(loss="binary_crossentropy", optimizer=gen_optimizer)
discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)
```

We are using `Adam`optimizer as optimization algorithm and using `binary_crossentropy`as loss function. `Adam`Specify the optimizer's hyperparameter values ​​in the first step .  

3. Then, create and compile the adversarial model:
```python
discriminator.trainable = False adversarial_model = Sequential()
adversarial_model.add(generator)
adversarial_model.add(discriminator)  adversarial_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=gen_learning_rate, beta_1=beta))
```

4. After that, extract and load all ***airplane*** images for training:
```python
def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']   
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))   
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2),mode='constant', order=0)   
    return voxels
    
def get3ImagesForACategory(obj='airplane', train=True, cube_len=64, obj_ratio=1.0):
	obj_path = DIR_PATH + obj + '/30/'
	obj_path += 'train/' if train else 'test/'
	fileList = [f for f in os.listdir(obj_path) if f.endswith('.mat')]
	fileList = fileList[0:int(obj_ratio * len(fileList))]
	volumeBatch = np.asarray([getVoxelsFromMat(obj_path + f, cube_len) for f in fileList], dtype=np.bool)
	return volumeBatch
	
volumes = get3ImagesForACategory(obj='airplane', train=True, obj_ratio=1.0)
volumes = volumes[..., np.newaxis].astype(np.float)
```

5. Next, add ***Tensorboad*** callback and add ***generator*** and ***discriminator*** network.

```python
tensorboard = TensorBoard(log_dir="/".format(log_dir, time.time()))
tensorboard.set_model(generator)
tensorboard.set_model(discriminator)
```

6. Add a loop that will run for a specified number of cycles.
```python
for epoch in range(epochs):
    print("Epoch:", epoch)
 
## Create two lists to store losses
    gen_losses = []
    dis_losses = []
```

7. Add another loop within the first loop to run the specified batch size: 
```python
    number_of_batches = int(volumes.shape[0] / batch_size)
    print("Number of batches:", number_of_batches)
    for index in range(number_of_batches):
        print("Batch:", index + 1)
```

8. Next, a batch of images is sampled from a set of real images, and a batch of noise vectors are sampled from a Gaussian normal distribution. The shape of the noise vector should be `(1, 1, 1, 200)`:
```python
z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, 
                            z_size]).astype(np.float32)
volumes_batch = volumes[index * batch_size:(index + 1) * batch_size, 
                        :, :, :]
```

9. Generating fake images using a generator network. Pass it `z_sample`a batch of noise vectors and generate a batch of fake images:
```python
gen_volumes = generator.predict(z_sample,verbose=3)
```

10. Next, the discriminator network is trained on the fake images generated by the generator and a batch of real images from a set of real images. Additionally, make the discriminator easy to train:

```python
## Make the discriminator network trainable discriminator.trainable = True
 
## Create fake and real labels
labels_real = np.reshape([1] * batch_size, (-1, 1, 1, 1, 1))
labels_fake = np.reshape([0] * batch_size, (-1, 1, 1, 1, 1))
 
## Train the discriminator network
loss_real = discriminator.train_on_batch(volumes_batch, 
                                         labels_real)
loss_fake = discriminator.train_on_batch(gen_volumes, 
                                         labels_fake)
 
## Calculate total discriminator loss
d_loss = 0.5 * (loss_real + loss_fake)
```

The preceding code trains the discriminator network and calculates the total discriminator loss.  
11. Train an adversarial model that includes both the `generator`and `discriminator`network:
```python
z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
 
## Train the adversarial model        
g_loss = adversarial_model.train_on_batch(z, np.reshape([1] * batch_size, (-1, 1, 1, 1, 1)))
```

Additionally, add the losses to their respective lists as follows:
```python
gen_losses.append(g_loss)
dis_losses.append(d_loss)
```

12. Generate and save 3D images every other cycle:
```python
 if index % 10 == 0:
            z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            generated_volumes = generator.predict(z_sample2, verbose=3)
            for i, generated_volume in enumerate(generated_volumes[:5]):
                voxels = np.squeeze(generated_volume)
                voxels[voxels < 0.5] = 0.
  voxels[voxels >= 0.5] = 1.
  saveFromVoxels(voxels, "results/img___".format(epoch, index, i))
```

13. After each epoch, save the average loss to `tensorboard`:

```python
## Save losses to Tensorboard write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)
```

My suggestion is to train it for 100 epochs to find problems in your code. After solving these problems, you can train the network for 100,000 epochs.

## Save model

After training model, just save model weights of generator and discriminator:
```python
""" Save models """ 
generator.save_weights(os.path.join(generated_volumes_dir, "generator_weights.h5"))
discriminator.save_weights(os.path.join(generated_volumes_dir, "discriminator_weights.h5"))
```

## Test model

To test the network, create `generator`and `discriminator`network. Then, load the learned weights. Finally, `predict()`generate predictions using the method:

```python
## Create models generator = build_generator()
discriminator = build_discriminator()
 
## Load model weights generator.load_weights(os.path.join(generated_volumes_dir, "generator_weights.h5"), True)
discriminator.load_weights(os.path.join(generated_volumes_dir, "discriminator_weights.h5"), True)
 
## Generate 3D images z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
generated_volumes = generator.predict(z_sample, verbose=3)
```

In this section, we successfully trained the generator and discriminator of 3D-GAN. In the next section, we will explore hyperparameter tuning and various hyperparameter optimization options.

## Visualize loss

To visualize the loss of training, start `tensorboard`the server as follows.
```python
tensorboard --logdir=logs
```


# Hyperparameter Optimization

The model we train may not be a perfect model, but we can optimize the hyperparameters to improve it. There are many hyperparameters in 3D-GAN that can be optimized. These include:

- **Batch size** : Try using a batch size value of 8, 16, 32, 54, or 128.
- **Number of cycles** : Try 100 cycles and gradually increase it to 1,000-5,000.
- **Learning rate** : This is the most important hyperparameter. Experiment with 0.1, 0.001, 0.0001 and other smaller learning rates.
- Activation functions in different layers of **generator and discriminator networks : Experiment with Sigmoid, tanh, ReLU, LeakyReLU, ELU, SeLU and other activation functions.**
- **Optimization Algorithms** : Try using Adam, SGD, Adadelta, RMSProp and other optimizers available in the Keras framework.
- **Loss function** : Binary cross entropy is the most suitable loss function for 3D-GAN.
- **Number of layers in both networks** : Depending on the amount of training data available, try different numbers of layers in the network. You can make the network deeper if you have enough data available for training.
# Practical applications of 3D-GAN

3D-GAN can be widely used in various industries, as shown below:

- **Manufacturing** : 3D-GAN can be an innovative tool to help create prototypes quickly. They can come up with creative ideas and can help simulate and visualize 3D models.
- **3D printing** : The 3D images generated by 3D-GAN can be used to print objects in 3D printing. The manual process of creating 3D models is lengthy.
- **Design Process** : 3D generated models provide a good estimate of the end result of a specific process. They can show us what is going to be built.
- **New samples** : Similar to other GANs, 3D-GAN can generate images to train supervised models.
# Sources
[1] http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip