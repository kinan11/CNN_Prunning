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


def modified_kernel_density_estimator(x, data, h, s):
    m, d = data.shape

    kde_values = np.zeros(m)

    for i in range(m):
        for j in range(m):
            kernel_vals = 1
            for k in range(d):
                diff = x[i, k] - data[j, k]
                scaled_diff = diff / (h[k] * s[j])
                kernel_vals *= gaussian_kernel(scaled_diff)
            kde_values[i] += kernel_vals / (s[j] ** d)
        kde_values[i] = kde_values[i] / (m * np.prod(h))

    return kde_values

def kernel_density_gradient(x_new, data, h, s):
    gradient = []
    X = data.T
    n, m = X.shape
    norm_const = (2 * np.pi) ** (-n / 2) / np.prod(h)

    for g in range(m):
        x = x_new[g]
        grad = np.ones(n)

        for dim in range(n):
            gradient_sum = 0
            for i in range(m):
                prod = 1
                for j in range(n):
                    u = (x[j] - X[j, i]) / (h[j] * s[i])
                    if j == dim:
                        prod *= np.exp(-u ** 2 / 2) * (-1) * u / (h[j] * s[i])
                    else:
                        prod *= np.exp(-u ** 2 / 2)
                gradient_sum += prod * (1 / s[i]) ** n
            grad[dim] = norm_const * gradient_sum / m
        gradient.append(grad)

    return gradient

def calculate_s(data, h):
    c = 0.5
    kde_values = standard_kernel_density_estimator(data, h)
    geometric_means = np.exp(np.mean(np.log(kde_values), axis=0))
    return (kde_values / geometric_means) ** (-c)

def single_modified_kernel_density_estimator(x, data, h, s):
    m, d = data.shape

    kde_values = 0

    for j in range(m):
        kernel_vals = 1
        for k in range(d):
            diff = x[k] - data[j, k]
            scaled_diff = diff / (h[k] * s[j])
            kernel_vals *= gaussian_kernel(scaled_diff)
        kde_values += kernel_vals / (s[j] ** d)
    return kde_values/ (m * np.prod(h))

def x_d_estimator(x, X, h, s):
    m = X.shape[0]
    estym = 0

    for i in range(m):
        x1 = (x[0]- X[i][0]) / (h[0] * s[i])
        x2 = (x[0] + X[i][0]) / (h[0] * s[i])
        tmp = (np.exp(-0.5 * x1**2) + np.exp(-0.5 * x2**2))
        estym += tmp / (s[i])

    f = estym / h / m * (2 * np.pi)**(-1 / 2)
    return f

def calculate_s_x_d(x, h):
    m = x.shape[0]
    c = 0.5
    s = np.ones(m)
    est = np.array([x_d_estimator(x[i], x, h, s) for i in range(m)])

    geo_mean = np.exp(np.mean(np.log(est)))

    s = (est ** -c) / (geo_mean ** -c)

    return s