import math

import numpy as np


def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def standard_kernel_density_estimator(data, h):
    m, d = data.shape

    kde_values = np.zeros(m)

    for i in range(m):
        for j in range(m):
            kernel_vals = 1
            for k in range(d):
                diff = data[i, k] - data[j, k]
                scaled_diff = diff / h[k]
                kernel_vals *= gaussian_kernel(scaled_diff)
            kde_values[i] += kernel_vals
        kde_values[i] = kde_values[i] / (m * np.prod(h))

    return kde_values


def modified_kernel_density_estimator(data, h, s):
    m, d = data.shape

    kde_values = np.zeros(m)

    for i in range(m):
        for j in range(m):
            kernel_vals = 1
            for k in range(d):
                diff = data[i, k] - data[j, k]
                scaled_diff = diff / (h[k] * s[j])
                kernel_vals *= gaussian_kernel(scaled_diff)
            kde_values[i] += kernel_vals / (s[j] ** d)
        kde_values[i] = kde_values[i] / (m * np.prod(h))

    return kde_values


# def gradient_kernel_density_estimator(data, h, s):
#     m, d = data.shape
#
#     kde_values = np.zeros((m, d))
#
#     for i in range(m):
#         kernel_vals = np.zeros(d)
#         for j in range(m):
#             for k in range(d):
#                 diff = data[i, k] - data[j, k]
#                 scaled_diff = diff / (h[k] * s[j])
#                 kernel_vals[k] += (-1) * (data[i, k] - data[j, k]) / ((h[k] ** 2) * (s[j] ** 2)) * gaussian_kernel(scaled_diff)
#             kde_values[i] = np.add(kde_values[i], kernel_vals / s[j] ** d)
#         kde_values[i] = kde_values[i] / (m * np.prod(h))
#
#     return kde_values

def kernel_density_gradient(data, h, s):
    m, n = data.shape
    gradients = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            gradient_sum = 0.0
            for k in range(m):
                u = (data[i, j] - data[k, j]) / (h[j] * s[k])
                product = gaussian_kernel(u) * (-1) * (u / (h[j] * s[k]))
                gradient_sum += product / s[k]
            gradients[i, j] = gradient_sum / (m * h[j])
    return gradients

def gradient(data, h, s):
    grad = kernel_density_gradient(data, h, s)

    return grad


def kernel_density_estimator(data, h):
    c = 0.5

    kde_values = standard_kernel_density_estimator(data, h)

    geometric_means = np.exp(np.mean(np.log(kde_values), axis=0))
    s = (kde_values / geometric_means) ** (-c)

    modified_kde_values = modified_kernel_density_estimator(data, h, s)
    # modified_kde_values = kde_values

    return modified_kde_values, s
