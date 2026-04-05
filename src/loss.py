import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def loss(self, theta, X, y, l1=0, l2=0):
        pass
    @abstractmethod
    def gradient(self, theta, X, y, l1=0, l2=0):
        pass

class Quadratic(LossFunction):
    def loss(self, theta, X, y, l1=0, l2=0):
        m = len(y)
        y_hat = X @ theta
        r = y_hat - y
        loss = (1/m) * np.sum(r ** 2)

        #regularization
        if l1 > 0:
            loss += l1 * np.sum(np.abs(theta))
        if l2 > 0:
            loss += l2 * np.sum(theta ** 2)
        return loss

    def gradient(self, theta, X, y, l1=0, l2=0):
        m = len(y)
        y_hat = X @ theta
        r = y_hat - y

        gradient = (2/m) * (X.T @ r) # deltaJ(theta) = (2/m) * X^T * (X*theta - y) = (2/m) * X^T * r

        #regularization derivatives
        theta_no_bias = theta.copy()
        theta_no_bias[0] = 0 # not regularize bias
        if l1 > 0:
            gradient += l1 * np.sign(theta_no_bias)
        if l2 > 0:
            gradient += 2 * l2 * theta_no_bias
        return gradient



class Absolute(LossFunction):
    def loss(self, theta, X, y, l1=0, l2=0):
        m = len(y)
        y_hat = X @ theta
        r = y_hat - y
        loss = (1/m) * np.sum(np.abs(r))

        #regularization
        if l1 > 0:
            loss += l1 * np.sum(np.abs(theta))
        if l2 > 0:
            loss += l2 * np.sum(theta ** 2)
        return loss

    def gradient(self, theta, X, y, l1=0, l2=0):
        m = len(y)
        y_hat = X @ theta
        r = y_hat - y
        gradient = (1/m) * (X.T @ np.sign(r)) # deltaJ(theta) = (1/m) * X^T * sign(X*theta - y) = (1/m) * X^T * sign(r)
        
        #regularization derivatives

        theta_no_bias = theta.copy() 
        theta_no_bias[0] = 0 # not regularize bias 
        if l1 > 0:
            gradient += l1 * np.sign(theta_no_bias) 
        if l2 > 0:
            gradient += 2 * l2 * theta_no_bias
        return gradient