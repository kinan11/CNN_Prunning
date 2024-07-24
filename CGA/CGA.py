import numpy as np
from CGA.calaculate_h import calculate_list_of_h
from CGA.kernel_density_estimator import calculate_s, modified_kernel_density_estimator, kernel_density_gradient


def calculate_d(x):
    dist = 0
    for i in range(x.shape[0] - 1):
        for j in range(i+1, x.shape[0]):
            dist += np.linalg.norm(x[i] - x[j])
    return dist


def complete_gradient_algorithm(data):
    num_iterations = 500
    x = data.copy()
    h = calculate_list_of_h(x)
    b = ((np.power(h, 2)) / (data.shape[1] + 2))
    # b = np.power(h, 2) / 3
    d0 = calculate_d(data)

    s = calculate_s(x, h)
    alpha = 0.001

    for iteration in range(num_iterations):

        dk_prev = calculate_d(x)
        f_value_curr = modified_kernel_density_estimator(x, data, h, s)
        grad = kernel_density_gradient(x, data, h, s)

        for i in range(data.shape[0]):
            x[i] += b * (grad[i] / f_value_curr[i])

        dk = calculate_d(x)

        if abs(dk - dk_prev) <= alpha * d0:
            break
    print(iteration)
    return x, h, s
