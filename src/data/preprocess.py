import numpy as np


def compute_tanh_stats(df, norm_cols):
    means = {}
    stds = {}
    for col in norm_cols:
        means[col] = df[col].mean()
        stds[col] = df[col].std()
    return means, stds


def apply_tanh_normalization(df, norm_cols, means, stds):
    df = df.copy()
    for col in norm_cols:
        df[col] = 0.5 * (np.tanh(0.01 * ((df[col] - means[col]) / stds[col])) + 1)
    return df
