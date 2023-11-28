# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:10:49 2023

@author: berat
"""

import numpy as np
import matplotlib.pyplot as plt
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


num_examples = X.shape[0]

#weight and bias parameters random
W = 0.01 * np.random.randn(D, K) #random weights (2,3)
b = np.zeros((1,K)) # 1x3

#hyperparameters
step_size = 1
reg = 0.001   #regularization strength

losses = []
#train a softmax linear classifier
iteration_number = 10000 #any number
for i in range(iteration_number):
    
    #1.Compute class score via single matrix multiplication
    s = np.dot(X, W) + b # 300x3
    
    #2.Compute the loss : cross-entropy loss/softmax
    #unnormalized probs
    exp = np.exp(s)
    #normalized probs
    probs= exp/np.sum(exp, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y]) #log based 2
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    losses.append(loss)
    
    #3.) Compute the analytic gradient with backpropagation
    dS = probs
    dS[range(num_examples), y] -= 1 
    dS /= num_examples    
    
    
    dW = np.dot(X.T, dS)
    db = np.sum(dS, axis=0, keepdims=True)
    dW += reg*W #regularization gradient
    
    #4.) Perform parameter update
    W += -step_size * dW
    b += -step_size * db
    
#6.) Draw a graph with respect to loss change for each iteration
plt.figure(figsize=(8, 6))
plt.plot(range(iteration_number), losses)    
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Change for Each Iteration') 
plt.show()    
print(f"Lastly Loss = {loss}")

#7.) Calculate accuracy
score = np.dot(X, W) + b
predicted_class = np.argmax(score, axis=1)
accuracy = np.mean(predicted_class == y)
print(f"Training Accuracy = {accuracy}")
#Since it is not a data set that shows linear propagation, 
#our accuracy is low.

#8.) Plot the learned decision boundaries
plt.figure(figsize=(8, 6))
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.title('Learned Decision Boundaries')
plt.show()


