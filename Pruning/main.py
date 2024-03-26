import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from CGA.CGA import complete_gradient_algorithm
from CGA.calaculate_h import calculate_list_of_h
from CGA.cluster_CGA import cluster_algorithm
from CGA.kernel_density_estimator import gradient, kernel_density_estimator


def main():
    mean1 = [-1, 0]
    mean2 = [1, 0]
    mean3 = [0, -1]

    cluster1 = [np.random.normal(mean1) + [-10, -10] for _ in range(1)]
    cluster2 = [np.random.normal(mean2) + [10, 0] for _ in range(1)]
    cluster3 = [np.random.normal(mean3) + [0, 100] for _ in range(1)]

    data = np.concatenate((cluster1, cluster2, cluster3), axis=0)
    # data = np.concatenate((cluster1, cluster2), axis=0)

    np.random.shuffle(data)

#     iris = load_iris()
#     data = iris.data
#     # data = np.array([[1.1, ], [0.1], [2.1], [1.12], [1.7], [15.0]])
#     data = np.array([
#     [3, 6],
#     [3.3, 7],
#     [5, 4],
#     [31.3, 7],
#     [61.3, 7],
# ])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, h = complete_gradient_algorithm(data)

    x = scaler.fit_transform(x)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x[:, 0], x[:, 1], marker='o')
    plt.show()

    z = cluster_algorithm(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cluster_indices, color in zip(z, colors):
        for index in cluster_indices:
            ax.scatter(data[index, 0], data[index, 1], c=[color], marker='o')

    plt.show()


if __name__ == "__main__":
    main()