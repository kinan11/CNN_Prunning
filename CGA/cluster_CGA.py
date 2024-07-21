import math
import numpy as np
from CGA.calaculate_h import calculate_list_of_h
from CGA.kernel_density_estimator import calculate_s, single_modified_kernel_density_estimator


def calculate_distances(data):
    distances = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            distances.append(np.linalg.norm(data[i] - data[j]))

    return distances


def reduce_sample(odl):
    np.random.seed(3)
    bootstrap = 1000
    points = bootstrap
    new_odl = []

    for i in range(points):
        numer = np.random.randint(0, points)
        new_odl.append(odl[numer])
        odl = np.delete(odl, numer)
        points -= 1

    odl = np.array(new_odl)
    return odl


def calculate_x_d(data):
    distances = calculate_distances(data)
    max_d = max(distances)

    sigma = np.std(distances)
    print("Max_d: ", max_d)
    print("Avg distances: ", np.average(distances))
    print("Sigma: ", sigma)

    d = (math.floor(100 * max_d) - 1)
    if d <= 0:
        d = 1
    converted_d = np.array([[x * 0.01 * sigma] for x in range(math.floor(d))])

    if len(distances) > 1000:
        dst = (np.array([[x] for x in reduce_sample(distances)]))
    else:
        dst = (np.array([[x] for x in distances]))

    h = calculate_list_of_h(np.array(dst))
    s = calculate_s(dst, h)

    kern_cur = 0
    diff_cur = 0
    min_odl = 0

    for i in range(1, math.floor(d) - 1):
        diff_prev = diff_cur
        kern_prev = kern_cur
        kern_cur = single_modified_kernel_density_estimator(converted_d[i], dst, h, s)
        diff_cur = kern_cur - kern_prev
        if (diff_prev < 0) and (diff_cur > 0):
            min_odl = converted_d[i]
            break

    return min_odl
def calculate_distance(data):
    distances = np.zeros(data.shape[0] - 1)
    for i in range(1, data.shape[0]):
        distances[i - 1] = np.linalg.norm(data[0] - data[i])
    return distances


def cluster_algorithm(data):
    x_d = calculate_x_d(data)
    clusters = []
    cluster_indices = []
    original_indices = list(range(data.shape[0]))

    while data.shape[0] > 0:
        distances = calculate_distance(data)
        indexes = [0]
        for index, element in enumerate(distances):
            if element < x_d:
                indexes.append(index + 1)

        if indexes:
            cluster = [data[i] for i in reversed(indexes)]
            clusters.append(cluster)
            cluster_index = [original_indices[i] for i in reversed(indexes)]
            cluster_indices.append(cluster_index)

            data = np.delete(data, indexes, axis=0)
            original_indices = [i for j, i in enumerate(original_indices) if
                                j not in indexes]

    print(x_d, len(clusters))
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster_indices[i]}")
    return cluster_indices