import math


def calculate_k6(x):
    return 1 / (math.sqrt(2 * math.pi)) * (x**6 - 15 * x**4 + 45 * x**2 - 15) * math.exp(-0.5 * x**2)


def calculate_k4(x):
    return 1 / (math.sqrt(2 * math.pi)) * (x**4 - 6 * x**2 + 3) * math.exp(-0.5 * x**2)


def calculate_c6(data, h):
    m = len(data)
    s = 0
    for i in range(m):
        for j in range(m):
            s += calculate_k6((data[i] - data[j]) / h)
    return s / ((m ** 2) * (h ** 7))


def calculate_c4(data, h):
    m = len(data)
    s = 0
    for i in range(m):
        for j in range(m):
            s += calculate_k4((data[i] - data[j]) / h)
    return s / ((m ** 2) * (h ** 7))


def calculate_h(data):
    m = len(data)
    x1 = 0
    x2 = 0
    for i in range(m):
        x1 += data[i] ** 2
        x2 += data[i]

    v = x1 / (m - 1) - (x2 ** 2) / (m * (m - 1))
    sigma = math.sqrt(v)
    c8 = 105 / (32 * math.sqrt(math.pi) * (sigma ** 9))

    h2 = ((-2 * (-15 / math.sqrt(2 * math.pi))) / (1 * c8 * m)) ** (1/9)
    c6 = calculate_c6(data, h2)

    h1 = ((-2 * 3/math.sqrt(2 * math.pi)) / (1 * c6 * m)) ** (1/7)
    c4 = calculate_c4(data, h1)

    wu = 1 / (2 * math.sqrt(math.pi))
    # wu = 0.354
    h = (wu / (c4 * m)) ** (1/5)
    return h


def calculate_list_of_h(data):
    n = len(data)
    m = len(data[0])
    h = []
    for i in range(m):
        h.append(calculate_h([data[j][i] for j in range(n)]))
    return h
