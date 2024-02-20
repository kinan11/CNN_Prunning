import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from CGA import complete_gradient_algorithm
from cluster_CGA import cluster_algorithm
from kernel_density_estimator import gradient


def main():
    # iris = load_iris()
    # data = iris.data
    # data = np.array([[1.1, ], [0.1], [2.1], [1.12], [1.7], [3.0]])
    data = np.array([
    [1, 2, 6],
    [1.5, 1.8,9],
    [5, 8, 1],
    [8, 8, 0],
    [1, 0.6, 2],
    [9, 11, 1]
])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, h = complete_gradient_algorithm(data)

    cluster_algorithm(x, h)


if __name__ == "__main__":
    main()
