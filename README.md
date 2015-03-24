# Handwritten-Digits-Classification
A novel appraoch to classify handwritten digits and comparison with traditional methods.
MNIST dataset of handwritten digits is classified using Multilayer Perceptron Neural Network, Naive Bayes Classifier and a novel hybrid model - Bayes-Neural Network Classifier.

Introduction

Classifiers for MNIST handwritten digits dataset: Multi-layer perceptron neural network, naive bayes classifier and a hybrid-model of naive bayes and mlp-nn: "bayes-neural network" are implemented, analyzed and compared. Each pixel of the input image is used as features to the models.

Multi-layer Perceptron Neural Network

A single hidden layer model with sigmoid units is used. The weights of neural network are learnt using back-propagation algorithm and regularization is implemented to prevent over-fitting. Effects of various hyper-parameters of the neural network on its accuracy are also presented.

Naive Bayes Classifier

Each image pixel is considered as a feature and each label of the digit is considered as a class. The image is converted to binary by thresholding on appropriate value, thus each feature is treated as Bernoulli random variable. Conditional probability tables for each feature given a class P(F1,F2,...,Fn|C) based on likelihood values. Prior of R.V class is considered as uniform and therefore ignored. Therefore, probability of class given a set of features P(C|F1,F2..Fn) is assumed to be directly related only to P(F1,F2,...,Fn|C). Log-likelihood values are used to infer as the product of probabilities diminish.

Hybrid Bayer-Neural Network

The network is built on the motivation that intermediate layers or output layer of the neural network can be considered as the features extracted from the input. These features are then considered independent nodes of naive bayes network. 
The nodes of the neural network are considered as a continuous random variables which are discretized to a form Multinolli R.V with domain: [0,1,...9]. The continuous values lie in the range of -infinity to +infinity (before sigmoid is applied on the nodes), is translated by the minimum value of feature vector of image (to remove negative values), normalized, scaled to 10, and then floor of the value is taken to construct the desired random variable.
The connection between the bayes and neural network is tested both at hidden layer and output layer.

