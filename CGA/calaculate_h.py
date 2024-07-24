import math

import numpy as np

def calculate_k6(x):
    return 1 / np.sqrt(2 * np.pi) * (x**6 - 15 * x**4 + 45 * x**2 - 15) * np.exp(-0.5 * x**2)

def calculate_k4(x):
    return 1 / np.sqrt(2 * np.pi) * (x**4 - 6 * x**2 + 3) * np.exp(-0.5 * x**2)

def calculate_c6(data, h):
    m = len(data)
    total_sum = 0
    for i in range(m):
        for j in range(m):
            total_sum += calculate_k6((data[j] - data[i]) / h)
    return 1 / (m ** 2 * h ** 7) * total_sum


def calculate_c4(data, h):
    m = len(data)
    total_sum = 0
    for i in range(m):
        for j in range(m):
            total_sum += calculate_k4((data[j] - data[i]) / h)
    return 1 / (m ** 2 * h ** 5) * total_sum


def calculate_h(data):
    m = len(data)
    x1 = 0
    x2 = 0
    for i in range(m):
        x1 += data[i] ** 2
        x2 += data[i]

    v = (x1 / (m - 1)) - ((x2 ** 2) / (m * (m - 1)))
    sigma = math.sqrt(v)
    c8 = 105 / (32 * np.sqrt(np.pi) * sigma**9)

    h2 = ((-2 * (-15 / np.sqrt(2 * np.pi))) / (c8 * m)) ** (1/9)
    c6 = calculate_c6(data, h2)

    h1 = ((-2 * 3 / np.sqrt(2 * np.pi)) / (c6 * m)) ** (1/7)
    c4 = calculate_c4(data, h1)

    wu = 1 / (2 * np.sqrt(np.pi))
    h = (wu / (c4 * m)) ** (1/5)

    return h


def calculate_list_of_h(data):
    data = np.array(data)
    transposed_data = data.T
    h = [calculate_h(column) for column in transposed_data]
    return h
