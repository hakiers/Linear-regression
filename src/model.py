import numpy as np
from src.functions import BaseFunction
from src.loss import LossFunction

class LinearRegressionModel:
    def __init__(self, base_function: BaseFunction, loss_function: LossFunction):
        self.base_function = base_function
        self.loss_function = loss_function
        self.theta = None
        self.history = []

    def fit_analytical(self, X, y, l1=0, l2=0):
        X_transformed = self.base_function.transform(X)
        n_features = X_transformed.shape[1]

        I = np.eye(n_features)
        I[0, 0] = 0 # not regularize bias
        #(X^T * X + lambda * I)^(-1) * X^T * y
        self.theta = np.linalg.inv(X_transformed.T @ X_transformed + l2 * I) @ X_transformed.T @ y
        return self.theta

    def fit_gradient_descent(self, X, y, alpha=0.01, max_iter=1000, tol=1e-6, l1=0, l2=0):
        X_transformed = self.base_function.transform(X)
        n_samples, n_features = X_transformed.shape
        self.theta = np.zeros(n_features)
        self.history = []

        for i in range(max_iter):
            gradient = self.loss_function.gradient(self.theta, X_transformed, y, l1, l2)
            theta_old = self.theta.copy()
            self.theta -= alpha * gradient

            ## Store history for visualization
            current_loss = self.loss_function.loss(self.theta, X_transformed, y, l1, l2)
            mse_value = self.mse(X, y)
            self.history.append((i, current_loss, mse_value))
            ###


            if np.linalg.norm(self.theta - theta_old) < tol:
                break
        return self.theta

    def fit_minibatch(self, X, y, alpha=0.01, max_iter=1000, tol=1e-6, batch_size=32, l1=0, l2=0):
        X_transformed = self.base_function.transform(X)
        n_samples, n_features = X_transformed.shape
        self.theta = np.zeros(n_features)
        self.history = []

        for i in range(max_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_transformed[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                gradient = self.loss_function.gradient(self.theta, X_batch, y_batch, l1, l2)
                theta_old = self.theta.copy()
                self.theta -= alpha * gradient
            
            #Store history for visualization
            current_loss = self.loss_function.loss(self.theta, X_transformed, y, l1, l2)
            mse_value = self.mse(X, y)
            self.history.append((i, current_loss, mse_value))
            ###

            if np.linalg.norm(self.theta - theta_old) < tol:
                break
        return self.theta

    def predict(self, X):
        X_transformed = self.base_function.transform(X)
        return X_transformed @ self.theta

    def mse(self, X, y):
        y_hat = self.predict(X)
        return np.mean((y_hat - y) ** 2)