---
title: Long Homework 1
shorttitle: Long Homework 1
notebook: Long_HW_1.ipynb
noline: 1
layout: wiki
---

## AMPTH 207: Stochastic Methods for Data Analysis, Inference and Optimization
 
**Due Date:** Thursday, Febrary 23rd, 2017 at 11:59pm

**Instructions:**

- Upload your final answers as well as your iPython notebook containing all work to Canvas.

- Structure your notebook and your work to maximize readability.

## Premise

In supervised machine learning, a function that maps certain input data to a set of outputs is inferred from a labelled dataset called the training set, and the resulting learnt funtion is then used to make predictions on unseen new examples of the dataset (the test set).

The goal of this homework is to construct a classifier known as the single-hidden-layer Multi-Layer Perceptron (MLP), an artificial neural netwok. You are asked to train the classifier using mini-batch gradient descent, validate it, and then apply it to a test dataset to make predictions. We will use the [*MNIST* dataset](https://en.wikipedia.org/wiki/MNIST_database), which consists of 70,000 images of handwritten digits, each of which is 28x28 pixels. You can use the first 50,000 images as the training set, the next 10,000 images as the validation set, and the last 10,000 images as the test set.

You will proceed in 2 steps (each step is a problem) to build the MLP. Please use Theano to program the classifiers.

## Problem 1. Stochastic gradient descent for the logistic regresion

First, build a logistic regression classifier whose input is the array of pixel values in one image, and whose output is the most likely class of the vector. In order to familiarize yourself with the dataset, plot some of the images beforehand, and think about the pixel values as features of the input vector.

### Part A

Using the softmax formulation, write a Theano expression graph that:
* Calculates the probability of a target element belonging to class $i$ (i.e., the probability that a given image represents a digit between 0 and 9).
* Maximizes it over all classes, and computes the cost function using an L2 regularization approach
* Minimizes the resulting cost function using mini-batch gradient descent. How long does it take for your code to train with 50,000 training examples of the dataset?

*Hint: Use a batch size of 256 examples, a learning rate $\eta = 0.1$, and a regularization factor $\lambda=0.01$*.

### Part B

* Evaluate the validation loss function periodically as you train the classifier and plot it as a function of the epoch. Plot this loss function for several values of the regularization factor.
* When should you stop the training for different values of $\lambda$? Give an approximate answer supported by using the plots.
* Select what you consider the best regularization factor and predict the classes of the test set. Compare with the given labels. What is the test error that you obtain?

## Problem 2. The Multilayer Perceptron (with one hidden layer)
The multilayer perceptron can be understood as a logistic regression classifier in which the input is first transformed using a learnt non-linear transformation. The non-linear transformation is usually chosen to be either the logistic function or the $\tanh$ function, and its purpose is to project the data into a space where it becomes linealry separable The output of this so-called hidden layer is then passed to the logistic regression graph that we have constructed in the first problem. In matrix notation:

$$G(b^{(2)}+W^{(2)}(s(b^{(1)}W^{(1)}x)))$$

with bias vectors $b^{(1)}$, $b^{(2)}$; weight matrices $W^{(1)}$, $W^{(2)}$ and activation functions $G$ and $s$. Here is a diagram:

![](http://deeplearning.net/tutorial/_images/mlp.png){:height=300 width=300}

### Part A

Using a similar architecture as in the first part and the same MNIST dataset, built a Theano graph for the multilayer perceptron, using the $\tanh$ function as the non-linearity. Use $\eta = 0.1$ and $\lambda = 0.001$. Experiment with the batch size (use 20, 50, and 100 examples) and the number of units in your hidden layer (use 50, 100, and 200 units). For what combination of these parameters do you obtain the smallest value of the validation loss function after 50 epochs?

### Part B

Stop the trainning at a convenient validation loss and use the trained classifier to predict for the test set. How much better is your test error as compared to the logistic regression classifier?

*Hint 1:* The initialization of the weights matrix for the hidden layer must assure that the units (neurons) of the perceptron operate in a regime where information gets propagated. For the $\tanh$ function, it is advisable to initialize with the interval $[-\sqrt{\frac{6}{fan_{in}+fan_{out}}},\sqrt{\frac{6}{fan_{in}+fan_{out}}}]$, where $fan_{in}$ is the number of units in the $(i-1)$-th layer, and $fan_{out}$ is the number of units in the i-th layer.

*Hint 2:* You should feel free to get inspiration from [these tutorials](http://deeplearning.net/tutorial/mlp.html). However, we expect you to write your own code, inspired by the architecture shown in the lab.






```python
# To read the dataset
dataset='mnist.pkl.gz'
with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
```

