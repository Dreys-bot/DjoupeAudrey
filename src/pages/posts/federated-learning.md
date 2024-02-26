---
layout: '@/templates/BasePost.astro'
title: Federated Learning
description: This article introduces the concept of federated learning, a recent machine learning approach aimed at preserving the privacy of data used to train models. Unlike traditional centralized learning where data is aggregated in a single location, federated learning allows models to be trained directly on users' devices, without collecting or sharing the actual data.
pubDate: 2022-06-03T00:00:00Z
imgSrc: '/assets/images/FederatedLearning/work_FL.png'
imgAlt: 'Image post 4'
---




# Abstract
Federated learning is a machine learning approach that allows multiple devices or systems to collaboratively train a machine learning model without the need to share their raw data with each other. In traditional centralized machine learning approaches, data is collected and aggregated into a central location before a
model is trained. However, in federated learning, the model is trained locally on each device or system using its
own data, and the updated model parameters are then sent to a central server, where they are aggregated to
create a new model that is then distributed back to all the devices.

This approach offers several advantages, such as preserving data privacy by keeping data local and not sharing
it across multiple systems. This can be especially important in industries such as healthcare, finance, or telecommunications where data privacy is critical. Federated learning can also be more efficient and scalable,
as it reduces the need for data transfer and can enable large-scale collaborations without requiring the transfer
of large amounts of data.

There are also some challenges with federated learning, such as ensuring the quality and consistency of the
data, dealing with imbalanced data distribution across devices, and handling communication and security
issues. However, as the technology continues to develop, it has the potential to revolutionize machine learning
by enabling large-scale collaborations and preserving data privacy.

**Keywords**: Decentralization, Privacy, Collaborative Learning, Data Security, Model Aggregation


# Introduction

Federated learning (1) is a machine learning technique that allows multiple parties to collaboratively
train a machine learning model without sharing their private data. In traditional machine learning, all data is
collected and centralized in a single location, such as a server or data center, where the model is trained.
However, in federated learning, the data remains decentralized and is stored locally on devices such as
smartphones or IoT devices.

In federated learning, the model is initially trained on a small sample of data from each device, and the
updated model is then sent back to the devices to improve its accuracy. Each device locally trains the model
using its private data and sends the updated model weights to the central server. The server then aggregates
the updates and uses them to improve the model. This process is repeated iteratively until the desired level of
accuracy is achieved.

One of the key advantages of federated learning is that it allows organizations to collaborate and train
machine learning models on a large amount of data without the need to centralize or share their data,
preserving data privacy and security. Federated learning is particularly useful in scenarios where the data is
sensitive, such as healthcare, finance, and personal devices.
Federated learning has a wide range of applications, including personalized recommendation systems,
natural language processing, image and video recognition, and predictive maintenance. However, there are
also some challenges associated with federated learning, such as communication and computational costs, as
well as the risk of biased or inaccurate models. Nonetheless, with ongoing research and advancements in
federated learning, it has the potential to revolutionize the way machine learning models are trained and
deployed in various industries.

# Architecture of federated learning

![architecture](/assets/images/FederatedLearning/architecture.png)


The architecture of federated learning (2) typically consists of three main components: the client
devices, the central server, and the machine learning model.

**Client devices**: The client devices are the endpoints that hold the local data and are used to train the machine learning model. These devices can be mobile phones, laptops, IoT devices, or any other device capable of running a machine learning algorithm. In federated learning, the data remains on the client devices, and the algorithm runs on each device locally.

**Central server**: The central server acts as a coordinator and aggregator for the training process. It is responsible for managing the training process, aggregating the model updates from the client devices, and sending the updated model back to the devices. The server can also perform additional tasks, such as initializing the model and distributing it to the client devices.

**Machine learning model**: The machine learning model is the algorithm used to learn from the data on the client devices. The model can be any type of supervised or unsupervised learning algorithm, such as neural networks, decision trees, or logistic regression.

# Types of federated learning

According to the distribution features of the data, federated learning may be categorized. Assume that the data matrix Di represents the information owned by each individual data owner, i.e., each sample and each characteristic are represented by a row and a column, respectively, in the matrix. At the same time, label data may be included in certain datasets as well. For example, we call the sample ID space I, the feature space X and the label space Y. When it comes to finance, labels may represent the credit of customers; when it comes to marketing, labels can represent the desire of customers to buy; and when it comes to education, labels can represent students' degrees. The training dataset includes the features X, Y, and IDs I. Federated learning may be classified as horizontally, vertically, or as federated transfer learning (FTL) depending on how the data is dispersed among the many parties in the feature and sample ID space. We cannot guarantee that the sample ID and feature spaces of the data parties are similar.

## Federated Transfer Learning (FTL)

![FTL](/assets/images/FederatedLearning/FTL.png)

Federated transfer learning is suitable while two datasets differ not only just in sample size but also in feature space. Consider a bank in China and an e-commerce firm in the United States as two separate entities. The small overlap between the user populations of the two institutions is due to geographical constraints. However, only a tiny fraction of the feature space from both companies overlaps as a result of the distinct enterprises. For example, transfer-learning may be used to generate solutions of problems for the full dataset and features under a federation. Specifically, a typical portrayal across the 2 feature spaces is learnt by applying restricted general sample sets as well as then used to produce prediction results for samples with just one-sided features. There are difficulties that FTL addresses that cannot be addressed by current federated learning methods, which is why it is an essential addition to the field.

$X_i \neq X_{j^{\prime}} Y_i \neq Y_{j^{\prime}} I_i \neq I_j \forall D_{i^{\prime}} D_{j^{\prime}}, i \neq j$


**Security Definition for FTL**: Two parties are normally involved in a federated transfer learning system. Due to its protocols' being comparable to vertical federated learning, the security definition for vertical federated learning may be extended here, as will be illustrated in the next.

### Vertical Federated Learning
![VFL](/assets/images/FederatedLearning/VFL.png)
Machine-learning techniques for vertically partitioned data have been suggested that preserve privacy, including gradient descent , classification , secure linear regression , association rule mining [46], and cooperative statistical analysis. Authors of  have presented a VFL method for training a logistic regression model that preserves individual privacy. Entity resolution and learning performance were investigated by the authors, who used Taylor approximation to approximate gradient and loss functions in order to provide homomorphic encryption for privacy-preserving computations. While 2 datasets share the same space of sample ID but vary in feature space, VFL, also known as feature-based FL (Figure 2(b)), may be used. An ecommerce firm and a bank are two examples of businesses in the same city that operate in quite different ways. The intersection of their user spaces is enormous since their user sets are likely to include most of the inhabitants in the region. Banks and e-commerce, on the other hand, keep track of their customers' income and spending habits and credit ratings; therefore, their feature sets are vastly different. Assume that both parties seek a product purchase prediction model based on product and user data. These distinct characteristics are aggregated, and the training loss and gradients are computed to develop a model that incorporates data from both parties jointly. In a federated learning system, every participating party has the same 12 identity and position, and the federal method helps everybody build a "common wealth" plan. 

**Security Definition of VFL**: Participant honesty and curiosity are assumed in a vertically federated-learning system. In a 2-party case, 2 parties are not collaborating, as well as only one is understood by an opponent. Only the corrupted client may learn data from the other client, and only that which the input or output disclose. Occasionally, a semi-honest third party (STP) is included to permit safe calculations between the parties, in which it is believed that the STP does not collude. These protocols are protected by SMC's formal privacy proof. At the conclusion of the learning process, every party has just the parameters of associated models with their own unique traits left in their memory bank. As a result, the two parties must work together at the inference stage to provide a result.

### Horizontal Federated Learning

![HTL](/assets/images/FederatedLearning/HFL.png)

For datasets that share a feature space but vary in the number of samples, sample-based federated learning or horizontal federated learning is presented. There is very little overlap between the user bases of two regional banks with very divergent customer bases. Nevertheless, due to the fact that their businesses are so similar, the feature spaces are identical. Collaboratively deep learning was suggested. Participants train individually and exchange just a limited number of parameter changes. Android phone model upgrades were suggested by Google in 2017 as a horizontal federated-learning approach. One user's Android phone changes the model parameters locally, then uploads them to the training the centralized model, Android cloud in concert with other owners of data. Their federated-learning approach is further supported by a safe aggregation technique that protects the aggregated user updates in privacy. To protect against the central server, model parameters are aggregated using additively homomorphic encryption. Several sites may execute independent tasks while maintaining security and sharing knowledge by using a multitask-style FL system. High fault tolerance stragglers, and communication costs difficulties may all be addressed by their multitask learning paradigm. It was suggested to establish a safe client–server structure where data is partitioned 13 by users as well as models developed on client devices work together to produce a global federated algorithm in the framework of an interconnected federated learning system. The model-building procedure prevents any data from leaking. Study looked at ways to reduce transmission costs so that data from remote mobile clients may be used in the training of centrally-located models. Deep gradient compression has recently been presented as a way to significantly decrease the bandwidth required for distributing large-scale training.


**Security Definition of HFL**: An honest-but-curious server is commonly assumed in a horizontally federated learning system. That is, only the server has the ability to intrude on the privacy of the data participants it hosts. These pieces of craftsmanship serve as evidence of their own safety. Additional privacy concerns have been raised by the recent proposal of a new security model that takes malevolent users into account. At the conclusion of the session, all of the parameters of the universal model are made available to each and every participant.


# Process of training
The federated learning process (3) typically follows the following steps:

**Initialization**: The machine learning model is initialized on the central server and distributed to the client
devices.

**Local training**: The client devices perform local training on their own data, using the machine learning
algorithm.

**Model update**: After local training, the client devices send their updated model parameters to the central
server.

**Aggregation**: The central server aggregates the model updates from all the client devices, using a specific
aggregation strategy, such as averaging or weighted averaging.

**Model distribution**: The updated model is distributed back to the client devices, and the process starts over
again.

Federated learning can also involve multiple rounds of training, where the local training and model
update steps are repeated multiple times before the final model is distributed. This process allows the model
to learn from a larger dataset and converge to a more accurate solution.

## How to process training?

The process of training a machine learning model (4) involves several steps, which can vary depending on the specific algorithm and data being used. However, a general overview of the process is as follows:

**Data preprocessing**: The first step in training a machine learning model is to preprocess the data. This can involve tasks such as cleaning the data, transforming it into a usable format, and splitting it into training and testing sets. 

**Model selection**: The next step is to select a machine learning algorithm that is suitable for the problem being addressed. This can involve evaluating the strengths and weaknesses of different algorithms, as well as considering factors such as model complexity, interpretability, and accuracy.

**Model initialization**: Once an algorithm has been selected, the model needs to be initialized with appropriate parameter values. This can involve randomly initializing the model parameters, or using a pre-trained model as a starting point. 

**Training**: The training process involves updating the model parameters to minimize the difference between the predicted outputs and the true outputs for the training data. This is typically done using an optimization algorithm such as stochastic gradient descent, which adjusts the model parameters based on the gradient of the loss function. 

**Validation**: During training, it is important to monitor the performance of the model on a validation set, which is a subset of the data that is not used for training. This can help to identify overfitting or underfitting, and allow for adjustments to the model. 

**Hyperparameter tuning**: Machine learning models often have hyperparameters, which are settings that are not learned during training but are set before training begins. These can include learning rate, regularization strength, and the number of hidden layers in a neural network. Tuning these hyperparameters can improve the performance of the model on the validation set. 

**Testing**: Once training is complete, the final model is evaluated on a separate testing set to estimate its generalization performance on new, unseen data. 

**Deployment**: The final step is to deploy the trained model in a production environment, where it can be used to make predictions on new data. This can involve integrating the model into a software system or deploying it as a web service.

# Tools for federated learning

There are several tools and frameworks available for implementing federated learning, some of which are:

**TensorFlow Federated**: TensorFlow Federated (TFF) is an open-source framework developed by Google that enables developers to implement federated learning using TensorFlow, a popular machine learning library. TFF provides a set of APIs for building and training federated learning models. 

**PySyft**: PySyft is an open-source framework developed by OpenMined that enables developers to implement privacy-preserving machine learning, including federated learning. PySyft provides a set of APIs for building and training federated learning models in Python. 

**Flower**: Flower is an open-source federated learning framework developed by Adap, which enables developers to build and train federated learning models using PyTorch. Flower provides a set of APIs for building and training federated learning models, as well as tools for managing federated learning workflows. 

**FedML**: FedML is an open-source framework developed by Tencent that provides a set of APIs for building and training federated learning models. FedML supports multiple machine learning frameworks, including TensorFlow, PyTorch, and Keras. 

**IBM Federated Learning**: IBM Federated Learning is a commercial product developed by IBM that provides a platform for building and training federated learning models. The platform supports multiple machine learning frameworks, including TensorFlow, PyTorch, and Keras.

**NVIDIA Clara Federated Learning**: NVIDIA Clara Federated Learning is a commercial product developed by NVIDIA that provides a platform for building and training federated learning models. The platform supports multiple machine learning frameworks, including TensorFlow, PyTorch, and Keras.

The choice of federated learning tools and frameworks will depend on factors such as the specific use case, the machine learning frameworks used, and the technical expertise of the development team.


