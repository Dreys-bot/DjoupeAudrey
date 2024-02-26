---
layout: '@/templates/BasePost.astro'
title: Dance style classification
description: Creation of a dance style classifier (afrobeat, hip-hop, classic.) from collected and processed videos, using convolutional neural networks on GPUs. Automatic extraction of key kinematic features and pose estimates via Mediapipe and AlphaPose. Deployment on a Web application for real-time classification.
githubPage: https://github.com/Dreys-bot/choreAI
pubDate: 2024-02-20T00:00:00Z
iconSrc: '/assets/icons/dance-icon.png'
iconAlt: 'Dance icon'
imgSrc: '/assets/images/paint-icon.png'
imgAlt: 'Paint icon'
tags: [Pytorch, Opencv, Alphapose, Tensorflow, Pandas]
---

# Abstract

Dance is a visual art form that uses the body and its movements as a means of expression. Being able to automatically analyze dance styles and techniques from videos would open up a wide range of pedagogical and cultural applications. In this project, i propose a method using pose estimation. It estimates 2D pose sequences, tracks dancers using AlphaPose with 17 key-body points from an online gathered dataset to create temporal motion sequences. An LSTM ecurrent neural network modelled the structure and numerous configurations were tested. Due to limit ressources, the model achieved an accuracy of 62% with a 24%.
Index Terms: Dance Style, Pose estimation, classification , LSTM, RNN

# Introduction 
With the rise of information technology and computer vision, human motion recognition has become an important research area with widespread applications. By leveraging video processing techniques, researchers have developed methods to extract and analyze motion features from video data to recognize different types of human actions. In particular, human pose estimation algorithms make it possible to capture the spatial configuration of the skeleton of individuals over time. This allows complete recognition of dance movements through the analysis of temporal sequences.

Although these technologies offer potential applications in areas such as dance teaching, current methods generally focus on human movement in general and very few have been applied to the recognition of specific dance styles. At the same time, the exponential growth of online video sharing platforms has created new opportunities and challenges for categorizing vast media content. YouTube alone supports over 500 hours of uploads per minute, making effective classification and tagging crucial for search optimization and visibility. However, user-generated tags may become biased or irrelevant over time.

This research aims to address these issues by automating the categorization of dance videos based on their presented style. By extracting pose sequences from uploaded content, dance movements can be analyzed and adapted to standardized forms. This allows objective, style-specific tags to be assigned to improve search and filtering. For dance communities and choreographers, it improves online exposure through optimized metadata. Basically, this work explores how computer vision and pattern recognition can help organize large online databases by parameterizing queries with dance style information.
# Dataset

Several large dance video datasets have been extracted from YouTube, such as UCF-101[1], Kinectics[2] and Atomic Visual Actions[3]. However, these studies do not include the Afrobeat dance style. Therefore, it was necessary to create a custom dataset for this project.

Videos were extracted from YouTube and Tiktok to constitute the dataset. Each of the three dance styles - Afrobeat, classical and hiphop - is represented by 393 videos, for a total of 1179 clips. YouTube provided a wide variety of styles and performers to choose from. Tiktok videos were also included as they tend to be brief in length.

Uniformity of the video characteristics was sought to ensure homogeneity of the corpus. Specifically, each video lasts approximately 25 seconds, allowing full movements or sequences to be captured. The chosen resolution is Full HD (1920x1080 pixels) to ensure optimal image quality.

As the videos are from public social media platforms, no legal constraints apply to their non-commercial use in research. Choosing YouTube as the primary source has the advantage of access to a large variety of dance styles and performers worldwide. This customized dataset aims to incorporate the important but previously omitted Afrobeat style for dance classification.

# Implementation
## Research Methodology

In my project, i focus on the creation of computer vision based model that is able to classify dance style: For this objective, i have set the following research questions:
- Which datasets exist and what features are needed or should be considered in order to perform such a research?
- What libraries and techniques exist to classsify a sequence of joint coordinated and match it with a dance style?
- How many dance styles are adequate to teach and test the algorithm?
## Dataset processing
Before any implementations, the first thing to do is to prepare the data to be used. Let's first recall the structure of our data:

- Videos
	1. Afrobeat
	2. Classical
	3.  hiphop
We have a folder for videos for each class. We must therefore process these videos to bring out the specific characteristics for each class. We know that each video is a succession of several images. So we divide the videos into images to highlight the dance steps in each video. So we have the following structure:
- images
	1. Afrobeat
	2. Classical
	3.  hiphop
For the rest of my project, I retained 84900 images of poses for each class with 100 timesteps. It means that each video has 100 frames, so we have 283 videos of each class( 28300frames per class) which is 70% of the total frames and 33000 frames for each class in test dataset, so we have 110 videos for each class which represent 30% of the total video.

So after preparing our data, we have the following structure:
- imagesSplit
	- train (28,300 images each)
		1. Afrobeat
		2. Classical
		3.  hiphop
	- test (33,000 images each)
		1. Afrobeat
		2. Classical
		3.  hiphop

## Tracking and 2D pose estimation

As our main goal is to classify style dance, we use mainly the estimation pose of videos. We use then **Alphapose** to retrieve all the pose estimation of our dataset. [AlphaPose](https://www.mvig.org/research/alphapose.html) is a state-of-the-art [human pose estimation](https://encord.com/glossary/human-pose-estimation-definition/) tool developed by the [Chinese Academy of Sciences](https://english.cas.cn/). It uses a deep learning algorithm to analyze images or videos and estimate the pose of one or multiple humans in real time. AlphaPose uses a [convolutional neural network](https://encord.com/glossary/cnn-definition/) (CNN) to estimate the pose of humans in images or videos. It analyzes the input image or video frame by frame, detecting human body parts such as the head, torso, and limbs. It then estimates the position and orientation of each body part, creating a pose estimation for the entire human body.

AlphaPose uses a bottom-up approach, which means it first detects individual body parts before estimating the overall pose. This allows it to simultaneously handle multiple humans in the same image or video. AlphaPose also uses a heatmap-based approach, which means it estimates the likelihood of each pixel belonging to a specific body part.

![[Pasted image 20240126073439.png]]

So after having all the frame of video, i extract pose of all person in the frame with AlphaPose with 17 2D key-body points. The pose results of each was saved in json file. To easy use them, i took each keypoint and put it on one line in a .txt file. Data was splitted into trainig and testing.

For our project, we use this command: 
```python
python demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --eval --gpu 0,1,2,3
```

| Parameters Name | Description |
| ---- | ---- |
| --cfg | Experiment configure file name |
| --checkpoint | Experiment checkpoint file name |
| --indir | Directory of input processed images |
| --gpu | Run model faster |
| --vis | to save pose estimation image |
This command runs the model inference on input images located in the specified directory. The experiment configuration and checkpoint files specify the model architecture and trained weights. Using multiple GPUs with the "--gpu" flag speeds up the inference.
### Output
A total of 51 body joints are detected by the AlphaPose model, including key points for head, shoulders, elbows, wrists, hips, knees and ankles. This output provides a clear visualization of all the detected keypoints on the input image.
```python
595.3865966796875, 206.50001525878906, 0.9428081512451172, 605.41259765625, 206.50001525878906, 0.9531299471855164, 600.3995971679688, 201.4870147705078, 0.9429413676261902, 620.4515380859375, 216.5260009765625, 0.9422314763069153, 600.3995971679688, 206.50001525878906, 0.7844216823577881, 620.4515380859375, 246.60398864746094, 0.8401278853416443, 595.3865966796875, 236.57798767089844, 0.8246780633926392, 575.3345947265625, 206.50001525878906, 0.7251867651939392, 570.3215942382812, 201.4870147705078, 0.7556909322738647, 590.3735961914062, 151.3570556640625, 0.7063803672790527, 585.360595703125, 146.34405517578125, 0.8612056374549866, 605.41259765625, 366.9158935546875, 0.8092090487480164, 585.360595703125, 366.9158935546875, 0.7450851798057556, 595.3865966796875, 472.1888122558594, 0.7834839224815369, 585.360595703125, 477.2018127441406, 0.8346507549285889, 605.41259765625, 577.4617309570312, 0.711732029914856, 600.3995971679688, 577.4617309570312, 0.7633700370788574
```

![[Pasted image 20240126075402.png]]

### Problems
I have some problem in this process:
- In my input directory, i have 84300 images, but the AlphaPose command cannot launch all this images once because of the capacity of my RAM and my computer. 
- The took many time without GPU

### Resolutions
- Run a bloc of images (2000 images) each time or use threading
- Use gpu to accelerate process

# Train dance style recognition

## Data Load and network input

The first step of the training network is to load dataset. We tell that we have 100 timesteps series in our video. We load it and create numpy bloc for each video. So the output is a numpy array of 100 lines of keypoints for one video and the label like in this code.

```python
def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)
    X_ = np.array(np.split(X_,blocks))  
    return X_
```

```python
print("X_train size", len(X_train))
print("X_test size", len(X_test))
print("y_train size", len(y_train))
print("y_test size", len(y_test))

Result

X_train size 849
X_test size 330 
X_train size 849 
X_test size 330
```


### Parameters of network

```python
training_data_count = len(X_train) #849 training series
test_data_count = len(X_test) #330 test series
n_input = len(X_train[0][0]) #num input parameters per timestep

  
n_hidden = 20 # Hidden layer num of features
n_classes = 3

#updated for learning rate decay
#calculated as:  decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

decaying_learning_rate = True
learning_rate = 0.00005 #used if decaying_learning_rate set to False
init_learning_rate = 0.0005
decay_rate = 0.96 #the base of the exponential in the decay
decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96

  
global_step = tf.Variable(0, trainable=False)
lambda_loss_amount = 0.0015

training_iters = training_data_count *200  # Loop 300 times on the dataset, ie 300 epochs
batch_size = 512
display_iter = batch_size*8  # To show test set accuracy during training
```

### Build LSTM Network

The proposed LSTM model effectively leverages the inherent sequential structure of body movements through a many-to-one architecture for dance style classification.

The LSTM_RNN function takes as input the poses extracted from videos in a form suitable for sequential processing by LSTM. The data is prepared by putting the temporal dimension in the first axis and is cut into tensors per time step.

Two stacked LSTM layers are specified to model long-term dependencies between successive poses in a recurrent manner. LSTM networks can have different architectures depending on the type of problem addressed and the input/output data:

- Many-to-one: the input is a sequence (here a video cut into poses) and the output is a single prediction for this sequence. This is the case here since we want to predict the dance style class for each analyzed video.
- Many-to-many: same input/output in sequence form. Not well suited for direct classification.
- One-to-one: input/output is a single example, without temporal dimension.

In our case, we exploit the temporal sequential dependencies between poses but want a prediction per analyzed video.

The many-to-one architecture therefore only keeps the last output of the LSTM layers after recurrently processing the entire input sequence. This contextual representation encapsulates the information of the video as a whole.  
It is then projected to class scores by a dense layer. Thus, the temporal dimension is efficiently exploited while obtaining an overall prediction per analyzed video.

```python
def LSTM_RNN(_X, _weights, _biases):

    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])  

    # Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)


    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.compat.v1.nn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
    lstm_last_output = outputs[-1]
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']
```


### Evaluation

The evaluation of the LSTM model has two key aspects: measuring the cost and optimizing the weights. The cost, also called the cost function, is calculated at each training iteration and measures the error between the model's predictions and the true labels. Here we use the softmax_cross_entropy_with_logits function which is well suited for classification problems. An L2 regularization term is also included in the cost in order to limit excessively high weight values and thus avoid overfitting. The Adam optimizer is used to stochastically minimize this cost over iterations, gradually fine-tuning the model weights. In addition, accuracy is calculated on the training and test data to evaluate the percentage of correct predictions. Furthermore, the learning rate decreases exponentially as a function of epochs, allowing the model to be finer-tuned more and more precisely during training. All of these evaluation and optimization mechanisms ensure the proper training of the LSTM model on the data while avoiding overfitting.

# Results

I retrieve two results. The first is the variation of accuracy with learning rate. We can notice that the high accuracy is 85% and it decreases when we downgrade the learning rate. The accuracy is highest is [1e-9, 1e-8, 1e-7].
![[Pasted image 20240215173557.png]]

![[Pasted image 20240215174122.png]]

# Future Work

Several avenues for improving and extending this work are envisioned for future research. First, I aim to estimate 3D poses rather than just 2D poses, which would provide additional spatial information useful for dance classification. In addition, combining music recognition from accompanying videos would help better contextualize dance movements.

It would also be relevant to further analyze movements by extracting other features besides just poses, such as speed, amplitude or fluidity of gestures. This could further improve the model's understanding of the subtleties specific to each dance style.

Finally, another interesting direction would be to generate new sequences of dance steps in different styles (afro, classical, hip-hop...) as output from the model, after training on large datasets. In the long term, this work could also be applied to the analysis and generation of movements in other domains than dance.

The avenues mentioned aim to deepen the study of movement sequences by adding new modalities (3D pose, music) and functionalities (generation, other classification criteria). This would help advance research on the analysis and synthesis of movements through deep learning.

# References

[1] K. Soomro, A. R. Zamir, and M. Shah, “Ucf101: A dataset of 101 human actions classes from videos in the wild,” ArXiv, vol. abs/1212.0402, 2012.
[2] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev, M. Suleyman, and A. Zisserman, “The kinetics human action video dataset,” CoRR, vol. abs/1705.06950, 2017. [Online]. Available: http://arxiv.org/abs/1705.06950
[3]  C. Gu, C. Sun, D. A. Ross, C. Vondrick, C. Pantofaru, Y. Li, S. Vi- jayanarasimhan, G. Toderici, S. Ricco, R. Sukthankar et al., “Ava: A video dataset of spatio-temporally localized atomic visual actions,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 6047–6056

