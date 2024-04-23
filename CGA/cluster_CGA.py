import math
import numpy as np
import statistics
import matplotlib.pyplot as plt

from CGA.calaculate_h import calculate_list_of_h
from CGA.kernel_density_estimator import kernel_density_estimator


def calculate_distances(data):
    distances = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            distances.append(np.linalg.norm(data[i] - data[j]))

    return distances


def calculate_x_d(data):
    distances = calculate_distances(data)
    max_d = max(distances)
    sigma = statistics.stdev(distances)  # odchylenie standardowe
    d = (math.floor(100 * max_d) - 1)
    if d <= 0:
        d = 1
    converted_d = np.array([[x * 0.01 * sigma] for x in range(math.floor(d))])
    h = calculate_list_of_h(converted_d)
    kde_d, s = kernel_density_estimator(converted_d, h)
    x_d = 1
    plt.scatter(range(len(kde_d)), kde_d)

    for i in range(1, math.floor(d) - 1):
        x_d = i
        if kde_d[i - 1] > kde_d[i] <= kde_d[i + 1]:
            print(i)
            break
    z = converted_d[x_d]
    plt.scatter(x_d, kde_d[x_d], color='red')
    plt.show()
    return converted_d[x_d]


def calculate_distance(data):
    distances = np.zeros(data.shape[0] - 1)
    for i in range(1, data.shape[0]):
        distances[i - 1] = np.linalg.norm(data[0] - data[i])
    return distances


def cluster_algorithm(data):
    x_d = calculate_x_d(data)
    clusters = []
    cluster_indices = []  # Lista do przechowywania indeksów klastrów
    original_indices = list(range(data.shape[0]))  # Lista indeksów oryginalnych

    while data.shape[0] > 0:
        distances = calculate_distance(data)
        indexes = [0]  # Indeksy elementów do klastra
        for index, element in enumerate(distances):
            if element < x_d:
                indexes.append(index + 1)

        if indexes:
            cluster = [data[i] for i in reversed(indexes)]
            clusters.append(cluster)
            cluster_index = [original_indices[i] for i in reversed(indexes)]  # Odpowiednie indeksy oryginalne
            cluster_indices.append(cluster_index)  # Dodajemy indeksy do listy klastrów

            data = np.delete(data, indexes, axis=0)
            original_indices = [i for j, i in enumerate(original_indices) if
                                j not in indexes]  # Aktualizacja oryginalnych indeksów

    print(x_d, len(clusters))
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster_indices[i]}")
    return cluster_indices