import numpy as np
from src.data import DataLoader
from src.scaler import StandardScaler
from src.model import LinearRegressionModel

def compute_learning_curves(model, fractions, n_runs=5, alpha=0.01, max_iter=1000, file_path=None):
    
    train_mse = {f: [] for f in fractions}
    test_mse = {f: [] for f in fractions}

    for run in range(n_runs):

        loader = DataLoader(file_path)
        loader.open_file()
        train_data, valid_data, test_data = loader.split()

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_valid, y_valid = valid_data[:, :-1], valid_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        for frac in fractions:
            n_train_samples = int(frac * len(X_train_scaled))
            X_train_frac = X_train_scaled[:n_train_samples]
            y_train_frac = y_train[:n_train_samples]

            model.fit_gradient_descent(X_train_frac, y_train_frac, alpha=alpha, max_iter=max_iter)
            train_mse[frac].append(model.mse(X_train_frac, y_train_frac))
            test_mse[frac].append(model.mse(X_test_scaled, y_test))

    avg_train_mse = [np.mean(train_mse[f]) for f in fractions]
    avg_test_mse = [np.mean(test_mse[f]) for f in fractions]

    return avg_train_mse, avg_test_mse