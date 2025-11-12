import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.neural_network import NeuralNetwork
from data.preprocess import compute_tanh_stats, apply_tanh_normalization


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns="target")
    y = df[["target"]]
    return X, y


def split_data(X, y, val_ratio=0.2, test_ratio=0.2, seed=44):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=seed
    )
    relative_val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val_size), random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(X_train, X_val, X_test, norm_cols):
    means, stds = compute_tanh_stats(X_train, norm_cols=norm_cols)
    X_train = apply_tanh_normalization(X_train.copy(), norm_cols, means, stds)
    X_val = apply_tanh_normalization(X_val.copy(), norm_cols, means, stds)
    X_test = apply_tanh_normalization(X_test.copy(), norm_cols, means, stds)
    return (
        X_train.to_numpy().T,
        X_val.to_numpy().T,
        X_test.to_numpy().T,
    )


def evaluate_model(model, X, y):
    y_pred, _, _ = model.forward_pass(X)
    y_pred_labels = (y_pred > 0.5).astype(int)
    return np.mean(y_pred_labels == y)


def train_and_select_model(X_train, y_train, X_val, y_val, layers, learning_rates):
    best_model = None
    best_acc = 0.0
    best_lr = 0.0

    for lr in learning_rates:
        print(f"\n=== Training with learning rate {lr} ===")
        model = NeuralNetwork(layers, X_train, y_train)
        _ = model.train(epochs=75000, learning_rate=lr)

        val_acc = evaluate_model(model, X_val, y_val)
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc, best_lr, best_model = val_acc, lr, model

    print("\n=== Best Model ===")
    print(f"Validation Accuracy: {best_acc * 100:.2f}%")
    print(f"Learning Rate: {best_lr}")
    return best_model


def main():
    layers = [13, 7, 1]
    norm_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    X, y = load_data("/Users/akildickerson/Projects/neural_network/data/raw/heart.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test, norm_cols)

    y_train = y_train.to_numpy().reshape(1, -1)
    y_val = y_val.to_numpy().reshape(1, -1)
    y_test = y_test.to_numpy().reshape(1, -1)

    best_model = train_and_select_model(
        X_train, y_train, X_val, y_val, layers, learning_rates
    )

    test_acc = evaluate_model(best_model, X_test, y_test)
    print("\n=== Test Accuracy ===")
    print(f"Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
