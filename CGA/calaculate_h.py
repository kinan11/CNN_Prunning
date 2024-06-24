import math

import numpy as np

def calculate_k6(x):
    return 1 / np.sqrt(2 * np.pi) * (x**6 - 15 * x**4 + 45 * x**2 - 15) * np.exp(-0.5 * x**2)

def calculate_k4(x):
    return 1 / np.sqrt(2 * np.pi) * (x**4 - 6 * x**2 + 3) * np.exp(-0.5 * x**2)

def calculate_c6(data, h):
    data = np.array(data)
    diff = (data[:, None] - data[None, :]) / h  # Oblicz macierz różnic
    return np.mean(calculate_k6(diff)) / (h**7)

def calculate_c4(data, h):
    data = np.array(data)
    diff = (data[:, None] - data[None, :]) / h  # Oblicz macierz różnic
    return np.mean(calculate_k4(diff)) / (h**7)


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

    # if h < 0.2:
    #     return 0.2
    return h


def calculate_list_of_h(data):
    n = len(data)
    m = len(data[0])
    h = []
    for i in range(m):
        h.append(calculate_h([data[j][i] for j in range(n)]))
    return h

# def calculate_list_of_h(data):
#     m = len(data[0])
#     h = []
#     for i in range(m):
#         h.append(0.03)
#     return h

# def calculate_list_of_h(data):
#     n, d = data.shape
#     bandwidths = []
#     for i in range(d):
#         std_dev = np.std(data[:, i])
#         bandwidth = 1.06 * std_dev * n ** (-1 / (d + 4))
#         bandwidths.append(bandwidth)
#     return np.array(bandwidths)