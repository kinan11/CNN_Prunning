import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Do 3D wykresów
from sklearn.preprocessing import StandardScaler

from CGA.CGA import complete_gradient_algorithm
from CGA.calaculate_h import calculate_h, calculate_list_of_h
from CGA.cluster_CGA import cluster_algorithm
from sklearn import datasets


def main():
    params = {
        'cluster_1': {'mean': [-15, -15, -15], 'cov': [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 'size': 500},
        'cluster_2': {'mean': [0, 0, 0], 'cov': [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 'size': 500},
        'cluster_3': {'mean': [15, 15, 15], 'cov': [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 'size': 500},
    }

    np.random.seed(0)
    cluster1 = np.random.multivariate_normal(params['cluster_1']['mean'], params['cluster_1']['cov'],
                                             params['cluster_1']['size'])
    cluster2 = np.random.multivariate_normal(params['cluster_2']['mean'], params['cluster_2']['cov'],
                                             params['cluster_2']['size'])
    cluster3 = np.random.multivariate_normal(params['cluster_3']['mean'], params['cluster_3']['cov'],
                                             params['cluster_3']['size'])

    data = np.concatenate((cluster1, cluster2, cluster3), axis=0)
    np.random.shuffle(data)
    # data = datasets.load_iris().data

    # data = np.array([[6], [7], [4]])
    # h = calculate_list_of_h(data)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o')
    plt.title('Oryginał')
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dimension_titles = ["Dimension 1", "Dimension 2", "Dimension 3"]

    for dim in range(3):
        axes[dim].scatter(range(data.shape[0]), data[:, dim], marker='o')
        axes[dim].set_title(dimension_titles[dim])
        axes[dim].set_xlabel("Index")
        axes[dim].set_ylabel(f"Dimension {dim + 1}")

    plt.suptitle("Oryginał - wymiary")
    plt.show()

    x, h = complete_gradient_algorithm(data)
    print("h: ", h)

    x = scaler.fit_transform(x)

    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']

    z = cluster_algorithm(x)

    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = 0
    for cluster_indices in z:
        for index in cluster_indices:
            ax.scatter(x[index, 0], x[index, 1], x[index, 2],  c=colors[i%6], marker='o')
        i += 1
    plt.title("Po CGCA")
    plt.show()


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dimension_titles = ["Dimension 1", "Dimension 2", "Dimension 3"]

    i = 0
    for cluster_indices in z:
        for index in cluster_indices:
            for dim in range(3):
                axes[dim].scatter(index, x[index, dim], c=colors[i % 6], marker='o')
        i += 1

    for dim, ax in enumerate(axes):
        ax.set_title(dimension_titles[dim])
        ax.set_xlabel("Index")
        ax.set_ylabel(f"Dimension {dim + 1}")

    plt.suptitle("Po CGCA")
    plt.show()


if __name__ == "__main__":
    main()
