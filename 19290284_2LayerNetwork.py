# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:10:49 2023

@author: berat
"""

#1.) This time use 2-layer neural network.

import numpy as np
import matplotlib.pyplot as plt

# Data Generation
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8')  # class labels

for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

 
#random params
h = 100                          #hidden layer size
W1 = 0.01*np.random.randn(D, h)  #(2x100)
b1 = np.zeros((1,h))             #(1x100)
W2 = 0.001*np.random.randn(h, K) #(100x3)
b2 = np.zeros((1,K))             #(1x3)

#hyperparameters
step_size = 1
reg = 0.001                      #regularization strength
iteration_number =10000
example_number = X.shape[0]
losses = []

for i in range(iteration_number):
    #Compute all class scores
    # max(0,z) is called ReLu activation function
    #ReLu is good default choice for an activation function
    #thresholded at 0
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # (300x100)
    s = np.dot(hidden_layer, W2) + b2                 # (300x3)
    
    #2.) Again use cross-entropy loss and backpropagation and do parameter update
    #class probs
    exp = np.exp(s)                                  # (300x3)
    probs = exp / np.sum(exp, axis=1, keepdims=True) # (300x3)
    
    #cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(example_number), y]) #log based 2
    data_loss = np.sum(correct_logprobs) / example_number
    reg_los = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_los
    losses.append(loss)
    #analytic gradient
    ds = probs
    ds[range(example_number),y] -= 1
    ds /= example_number
    #backpropagation
    dW2 = np.dot(hidden_layer.T, ds)
    db2 = np.sum(ds, axis=0, keepdims=True)
    d_hidden_layer = np.dot(ds, W2.T)
    d_hidden_layer[hidden_layer <= 0] = 0  #ReLU
    dW1 = np.dot(X.T, d_hidden_layer)
    db1 = np.sum(d_hidden_layer, axis=0, keepdims=True)
    dW2 += reg *W2
    dW1 += reg *W1
    
    #Perform parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2
    
#Draw a graph with respect to loss change for each iteration    
plt.figure(figsize=(8, 6))
plt.plot(range(iteration_number), losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss change over iterations')
plt.show()
print(f"Lastly Loss = {loss}")

# Calculate accuracy
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
pred_y = np.argmax(scores, axis=1)
accuracy = np.mean(pred_y == y)
print(f"Training Accuracy = {accuracy}")
#%96.6 , this can lead to overfitting #we can increase learning rate(step_size) 

# Plot the learned decision boundaries
plt.figure(figsize=(8, 6))
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.title('Learned Decision Boundaries')
plt.show()

