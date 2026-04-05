import numpy as np
from abc import ABC, abstractmethod

class BaseFunction(ABC):
    @abstractmethod
    def transform(self, X):
        pass

class Linear(BaseFunction):
    def transform(self, X):
        n = X.shape[0]
        ones = np.ones((n, 1))
        return np.hstack((ones, X)) # [1, x1, x2, ...]

class Polynomial(BaseFunction):
    def __init__(self, degree):
        self.degree = degree

    def transform(self, X):
        n, d = X.shape
        features = [np.ones((n, 1))] # [1]
        for i in range(1, self.degree + 1):
            features.append(X ** i) # [x1^i, x2^i, ...] 
        return np.hstack(features) # [1, x, x^2, ..., x^d]

class PolynomialInteraction(BaseFunction):
    def __init__(self, degree):
        self.degree = degree

    def transform(self, X):
        n, d = X.shape
        features = [np.ones((n, 1))]
    
        for i in range(self.degree + 1):
            for j in range(self.degree + 1):
                if i + j == 0:
                    continue
                if i + j <= self.degree:
                    term = (X[:, 0] ** i) * (X[:, 1] ** j)
                    features.append(term.reshape(-1, 1))
        return np.hstack(features) # [1, x1, x2, x1^2, x1*x2, x2^2, ...]

class Gaussian(BaseFunction):
    def __init__(self, sigma, n_centers=5):
        self.sigma = sigma
        self.n_centers = n_centers

    def transform(self, X):
        centers = np.linspace(np.min(X, axis=0), np.max(X, axis=0), self.n_centers)
        features = [np.ones((X.shape[0], 1))]

        for mu in centers:
            gauss = np.exp(-(X - mu) ** 2 / (2 * self.sigma ** 2))
            features.append(gauss)
        return np.hstack(features) # [1, exp(-(x-mu1)^2/(2*sigma^2)), exp(-(x-mu2)^2/(2*sigma^2)), ...]

class Sigmoid(BaseFunction):
    def __init__(self, sigma, n_centers=5):
        self.sigma = sigma
        self.n_centers = n_centers
    
    def transform(self, X):
        centers = np.linspace(np.min(X, axis=0), np.max(X, axis=0), self.n_centers)
        features = [np.ones((X.shape[0], 1))]

        for mu in centers:
            sigmoid = 1 / (1 + np.exp(-(X - mu) / self.sigma))
            features.append(sigmoid)
        return np.hstack(features) # [1, 1/(1+exp(-(x-mu1)/sigma)), 1/(1+exp(-(x-mu2)/sigma)), ...]

class Log(BaseFunction):
    def transform(self, X):
        n = X.shape[0]
        ones = np.ones((n, 1))
        return np.hstack((ones, np.log(np.abs(X) + 1))) # [1, log(x1), log(x2), ...]
        
class Sin(BaseFunction):
    def transform(self, X):
        n = X.shape[0]
        ones = np.ones((n, 1))
        return np.hstack((ones, np.sin(X))) # [1, sin(x1), sin(x2), ...]
    
class Cos(BaseFunction):
    def transform(self, X):
        n = X.shape[0]
        ones = np.ones((n, 1))
        return np.hstack((ones, np.cos(X))) # [1, cos(x1), cos(x2), ...]

class Fourier(BaseFunction):
    def __init__(self, n_freq):
        self.n_freq = n_freq

    def transform(self, X):
        n = X.shape[0]
        features = [np.ones((n, 1))]
        for k in range(1, self.n_freq + 1):
            features.append(np.sin(2 * np.pi * k * X))
            features.append(np.cos(2 * np.pi * k * X))
        return np.hstack(features) # [1, sin(2*pi*1*x), cos(2*pi*1*x), sin(2*pi*2*x), cos(2*pi*2*x), ...]

class Combined(BaseFunction):
    def __init__(self, bases):
        self.bases = bases

    def transform(self, X):
        results = []
        for i, basis in enumerate(self.bases):
            phi = basis.transform(X)
            if i == 0:
                results.append(phi)
            else:
                results.append(phi[:, 1:]) # skip bias
        return np.hstack(results)
