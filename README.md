# Capstone project -- Understand Neural Network from scratch
Author: Jiyao Li
Date: January 24th, 2017
Email: jiyao.lee@gmail.com

## Introduction
Neural Network (NN) is an import branch of Artificial Intelligence and Machine Learning. NN has capability of building and training a model to identify different complicated classes, which is not achievable with conventional linear/nonlinear classifiers, such as Softmax regression, SVM, etc. In addition, we all know that NN is a super hot topic nowadays.  

So I decided to study and explore Neural Network as my capstone project. However, after I tried several open-source NN packages, such as Caffe, Tensorflow, although they works quite well and fast, I still could not really understand how NN works, since these packages are all like a "black box" to me. Then, I started writing NN code from scratch and used several dataset to test my code. Based on the testing results, I think my code works. And through this process, I developed a much better understanding of the structure of NN and its features. 

I found several really good sources online to learn how to develop NN from scratch. The one I like the most is the class note of Stanford class CS231n, [Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/). This class teaches you how to build the basic architecture of a simple NN and then expand it to more and more complex structures like multi-layer NN and Convolutional Neural Network. Another good source is the [Deep Learning Tutorial](http://ufldl.stanford.edu/tutorial/) contributed by Andrew Ng and others. Actually, several major parts of my capstone project is based on the homework assignment 2 of the CS231n class. 


## Data 
There are three different datasets I used in this project. Among them, CIFAR-10 dataset is the one I mostly focus on, which is also the most complicated dataset. I also tested the simple one-layer and two-layer NN on the MNIST handwritten digits dataset. And to understand the basics of NN and its power over the other classification methods, e.g. logistic regression, I also created some simple synthetic dataset to test on.

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 32x32 3-color images in 10 classes. The whole dataset was split into 49,000 training examples, 1000 validation examples and 1000 testing examples. With fully-connected neural networks with 1 hidden layer or multiple hidden layers, even with some careful tuning of the hyper-parameters, validation and testing set accuracy can still be 55%. With my designed convolutional neural network (MineConvNet), accuracy for both validation set and testing set increase to 80%.

The [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/), has 70,000 32x32 gray-scale images in 10 classes. This dataset is split into 60,000 training examples, 10,000 validation examples. For this dataset, I did not spend much time tuning the hyper-parameters and designing the architecture of the neural net. With just one-layer or two-layer neural network, the testing accuracy can easily be over 90%.

Synthetic data, 2x2 (x and y coordinate) are either in two clusters or three clusters with labels. These clusters are separated from each other but some of them are linearly separable and some are not. This example is to demonstrate that Neural network has more power in correctly separating different classes in complicated situation.

I did not include the MNIST and CIFAR-10 dataset since they are too large. But it is easy to download them from the link I provided above.

## Code description
I wrote in Jupyter notebook for all the following sections and converted them into html file for the report. 
- LossGradientClassifiers.ipynb -> *Section 1* ```Python 3```
- FullyConnectedNets.ipynb -> *Section 2* ```Python 2```
-- Dropout.ipynb -> *Section 2.1* ```Python 2```
-- BatchNormalization.ipynb -> *Section 2.2* ```Python 2```
- ConvolutionalNetworks.ipynb -> *Section 3* ```Python 2```
-- CNN_Note_Convolution.ipynb -> *Section 3.1* ```Python 3```

I worked on the assignment of Stanford class CS231n and included some parts into this reports. So the cs231n folder has lots of facilitating code written by CS231n instructors. I wrote most of the functions in ```cs231n/layers.py```, ```cs231n/layer_utils.py```, and the code under ```cs231n/classifiers``` folder. ```layers.py``` has all the founding bricks for neural net. ```layer_utils.py``` has some convenient wrapper of this bricks to form founding blocks. ```cs231n/classifiers/fc_net.py``` is fully-connect Neural Net. ```cs231n/classifiers/cnn.py``` is Convolutional Neural Net. ```cs231n/classifiers/complex_cnn.py``` is the more complex CNN I build to finally train the CIFAR-10 dataset.

There is one setup to use Cython to make the CNN forward propagation and backpropagation faster rather than use plain Python. Please refer to **CS231n_assignment2_Setup_README.md** for the instructions. 

## Section 1 Basics of Logistic Regression, Softmax Regression, SVM and One-layer, Two-layer Neural Network
> Dataset: Synthetic dataset, MNIST dataset
> File: LossGradientClassifiers.html 

I summarized the differences between different classifiers, including their individual loss functions and gradient.

This section also demonstrates that the Neural Network has much more power separate different clusters of classes, which are not linear separable.  

The simple Neural Network is also applied to MNIST hand-writing digits dataset. The accuracy on the testing set can easily be over 90%. 


## Section 2 Fully connected neural network
> Dataset: CIFAR-10 dataset
> File: FullyConnectedNets.html 

This Section build more complex fully-connected Neural Network and train it with more complex dataset. After having the foundation of modules of forward propagation and backpropagation of a fully-connect layer, we can easily build more complicated neural network with arbitrary number of hidden layers.

This section also applies the Dropout and Batch normalization, which will be explained in the following two sub-sections.

> **Parameter updates in optimization**
> Several options are provided in the parameter update. We use mini-batch stochastic gradient descent methods.
> 
> Parameters update methods are:
> - Moment
> - Nesterov moment
> - Ada Grad
> - Ada Delta
> - RmsProp
> - Adam
> 
> Among them, "Adam" seems to be the best method for parameter update.
> 
> For more information about these parameter update methods, visit [CS231n class notes](http://cs231n.github.io/neural-networks-3/), [Sebastian Ruder blog](http://sebastianruder.com/optimizing-gradient-descent/), [Int8 blog](http://int8.io/comparison-of-optimization-techniques-stochastic-gradient-descent-momentum-adagrad-and-adadelta/#AdaDelta_8211_experiments).

### Section 2.1 Dropout
> Dataset: CIFAR-10 dataset
> File: Dropout.html

Dropout seems to be a "Mysterious" technique. In practice, dropout avoid overfitting and relying too much on some nodes by randomly ignoring them. [CS231n dropout section](http://cs231n.github.io/neural-networks-2/) shows how to do the dropout in practice.

### Section 2.2 Batch normalization and spatial batch normalization 
> Dataset: CIFAR-10 dataset
> File: BatchNormalization.html

Batch normalization normalize the input before the activation function to be zero mean and unit variance. In practice, using batch normalization will make the neural network less sensitive to value of weight scale used in weight initialization. [Clément thorey's blog](http://cthorey.github.io./backpropagation/) gives nice explanation of batch normalization. Actually, once you know the equation of batch normalization, it should be quite easy to derive the form of backpropagation (just use the chain rule).

## Section 3 Convolutional neural network
> Dataset: CIFAR-10 dataset
> File: ConvolutionalNetworks.html

Eventually, we come to the mysterious and powerful Convolutional Neural Network! After understanding the building blocks of CNN, I build a flexible CNN (MineConvNet class in cs231n/classifiers/complex_cnn.py), allowing user to expand the number of the Convolutional layers and number of the fully-connected layers. MineConvNet also included the functionality of dropout for the fully-connected layers, batch normalization and spatial batch normalization for the full-connected and convolutional layers respectively.

In the end, I build two CNN models with slightly different architectures to train the CIFAR-10 data set and use ensemble method to do prediction. The accuracy of the validation set and testing set are about 80%. This score is not even close to the best-score achieved on this dataset. However, it is a huge improvement from the fully-connected network. Because all of my code is in python (using numpy a lot), it is too too slow to train the complex models. It is definitely necessary to use GPU to train more complex Neural Network models. But I don't have the time and efforts at this moment to rewrite all these code into CUDA C. This is some future task for myself. 

### Section 3.1 CNN forward propagation and backpropagation
> File: CNN_Note_Convolution.html

After I understand the structure of normal Neural Network with just fully-connected layers, I was puzzled by Convolutional Neural Network. Why is it called "Convolutional", because convolution is a technique in signal or image processing. And how is the structure different from full-connect Neural Net and how to do forward and backward propagation with Convolutional Neural Network.