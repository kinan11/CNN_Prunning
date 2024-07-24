import math
import numpy as np
from CGA.calaculate_h import calculate_list_of_h
from CGA.kernel_density_estimator import calculate_s, single_modified_kernel_density_estimator, \
    x_d_estimator, calculate_s_x_d


def calculate_distances(data):
    distances = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            distances.append(np.linalg.norm(data[i] - data[j]))

    return distances


def reduce_sample(odl):
    np.random.seed(3)
    points = 2000
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

    if len(distances) > 2000:
        dst = (np.array([[x] for x in reduce_sample(distances)]))
    else:
        dst = (np.array([[x] for x in distances]))

    sigma = np.std(dst)
    print("Max_d: ", max_d)
    print("Avg distances: ", np.average(distances))
    print("Sigma: ", sigma)

    d = (math.floor(100 * max_d) - 1)
    if d <= 0:
        d = 1
    converted_d = np.array([[x * 0.01 * sigma] for x in range(math.floor(d))])

    h = calculate_list_of_h(np.array(dst))
    s = calculate_s_x_d(dst, h)

    f_prev = x_d_estimator(converted_d[0], dst, h, s)
    x_d = 0

    for i in range(1, len(converted_d) - 1):
        f_next = x_d_estimator(converted_d[i + 1], dst, h, s)
        f_curr = x_d_estimator(converted_d[i], dst, h, s)
        if f_prev > f_curr and f_curr <= f_next:
            x_d = converted_d[i]
            break
        f_prev = f_curr

    print("x_d: ", x_d)
    return x_d

def calculate_distance(data):
    distances = np.zeros(data.shape[0] - 1)
    for i in range(1, data.shape[0]):
        distances[i - 1] = np.linalg.norm(data[0] - data[i])
    return distances

def cluster_algorithm(data, h, s, data_old):
    x_d = calculate_x_d(data)
    return find_clusters(data, h, s, data_old, x_d)

def calculate_max_i(data, h, s, assigned_points):
    f_max = 0
    max_i = -1

    for i in range(data.shape[0]):
        if i in assigned_points:
            continue
        tmp = single_modified_kernel_density_estimator(data[i], data, h, s)
        if f_max < tmp:
            max_i = i
            f_max = tmp
    return max_i


def find_clusters(data, h, s, data_old, x_d):
    m = data_old.shape[0]
    clusters = []
    assigned_points = set()

    while len(assigned_points) < m:
        max_i = calculate_max_i(data_old, h, s, assigned_points)
        if max_i == -1:
            break

        current_cluster = [max_i]
        tmp = [max_i]
        assigned_points.add(max_i)

        for index in tmp:
            neighbors = np.where(np.linalg.norm(data - data[index], axis=1) < x_d)[0]
            for neighbor in neighbors:
                if neighbor not in assigned_points:
                    assigned_points.add(neighbor)
                    current_cluster.append(neighbor)
                    tmp.append(neighbor)

        clusters.append(current_cluster)

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {clusters[i]}")
    return clusters