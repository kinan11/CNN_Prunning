import math

import numpy as np

from CGA.calaculate_h import calculate_list_of_h
from CGA.kernel_density_estimator import kernel_density_estimator, gradient


def calculate_d(x):
    dist = 0
    for i in range(x.shape[0] - 1):
        for j in range(i+1, x.shape[0]):
            dist += np.linalg.norm(x[i] - x[j])
    return dist


def complete_gradient_algorithm(data):
    num_iterations = 1000
    x = data.copy()
    h = calculate_list_of_h(x)
    # h = [1, 1]
    b = (np.power(np.mean(h), 2)) / (data.shape[0] + 2)
    d0 = calculate_d(data)
    alpha = 0.001

    for iteration in range(num_iterations):
        dk_prev = calculate_d(x)
        f_value_curr, s = kernel_density_estimator(x, h)
        grad = gradient(x, h, s)
        for i in range(data.shape[0]):
            x[i] += b * (grad[i] / f_value_curr[i])

        dk = calculate_d(x)
        a = abs(dk - dk_prev)
        if abs(dk - dk_prev) <= alpha * d0:
            break

    print(iteration)
    return x, h
