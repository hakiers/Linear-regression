import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(model, title=''):
    if not model.history:
        return

    iterations = [h[0] for h in model.history]
    grad_norms = [h[1] for h in model.history]
    mse_values = [h[2] for h in model.history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(iterations, grad_norms, color='blue')
    ax1.set_title('Gradient Norm')
    ax1.set_xlabel('Iteration')
    ax1.grid(True)

    ax2.plot(iterations, mse_values, color='orange')
    ax2.set_title('MSE')
    ax2.set_xlabel('Iteration')
    ax2.grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--') 
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    plt.show()

def plot_regularization(alphas, mse_values, title=''):
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, mse_values, marker='o')
    plt.xscale('log')
    plt.xlabel('Regularization Strength')
    plt.ylabel('MSE')
    plt.title('MSE vs Regularization Strength')
    plt.grid(True)
    plt.suptitle(title)
    plt.show()


def plot_correlation_matrix(data):
    corr_matrix = np.corrcoef(data, rowvar=False)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.xticks(range(len(corr_matrix)), range(len(corr_matrix)), rotation=90)
    plt.yticks(range(len(corr_matrix)), range(len(corr_matrix)))
    plt.grid(False)
    plt.show()

def plot_learning_curves(train_mse, test_mse, fractions, title=''):
    plt.figure(figsize=(8, 5))
    plt.semilogx(fractions, train_mse, 'o-', label='Train MSE', color='blue')
    plt.semilogx(fractions, test_mse,  's-', label='Test MSE',  color='orange')
    for f, tr, te in zip(fractions, train_mse, test_mse):
        plt.annotate(f'{te:.0f}', (f, te), textcoords='offset points',
                     xytext=(5, 5), fontsize=8, color='orange')
    plt.xlabel('Training Set Fraction')
    plt.ylabel('MSE')
    plt.title(f'Learning Curves — {title}')
    plt.legend()
    plt.grid(True)
    plt.show()