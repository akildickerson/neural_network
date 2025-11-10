import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.neural_network import NeuralNetwork
from data.preprocess import compute_tanh_stats, apply_tanh_normalization


def main():
    layers = [13, 7, 1]
    norm_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]

    df = pd.read_csv("/Users/akildickerson/Projects/neural_network/data/raw/heart.csv")
    X = df.drop(columns="target")
    y = df[["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=44
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.33, random_state=44
    )

    X_train, X_val, X_test = normalize(X_train, X_val, X_test, norm_cols)

    y_train = y_train.to_numpy().reshape(1, len(y_train))
    y_val = y_val.to_numpy().reshape(1, len(y_val))
    y_test = y_test.to_numpy().reshape(1, len(y_test))

    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    best_model = None
    best_acc = 0.0
    best_lr = 0.0

    for lr in learning_rates:
        model = NeuralNetwork(layers, X_train, y_train)
        _ = model.train(epochs=75000, learning_rate=lr)
        val_pred, _, _ = model.forward_pass(X_val)
        val_pred_labels = (val_pred > 0.5).astype(int)
        accuracy = np.mean(val_pred_labels == y_val)
        print("========== VALIDATION ACCURACY ==========")
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        if accuracy > best_acc:
            best_acc = accuracy
            best_lr = lr
            best_model = model

    print("\n ========== BEST HYPER-PARAMETERS ==========")
    print(f"Best Validation Accuracy: {best_acc * 100:.2f}%")
    print(f"Best Learning Rate: {best_lr}")

    model = best_model
    y_pred, _, _ = model.forward_pass(X_test)

    y_pred_labels = (y_pred > 0.5).astype(int)

    accuracy = np.mean(y_pred_labels == y_test)
    print("\n ========== TEST ACCURACY ==========")
    print(f"Accuracy: {accuracy * 100:.2f}%")


def normalize(X_train, X_val, X_test, norm_cols):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    means, stds = compute_tanh_stats(X_train, norm_cols=norm_cols)
    X_train = apply_tanh_normalization(X_train, norm_cols, means, stds)
    X_val = apply_tanh_normalization(X_val, norm_cols, means, stds)
    X_test = apply_tanh_normalization(X_test, norm_cols, means, stds)

    X_train = X_train.to_numpy().T
    X_val = X_val.to_numpy().T
    X_test = X_test.to_numpy().T

    return X_train, X_val, X_test


if __name__ == "__main__":
    main()
