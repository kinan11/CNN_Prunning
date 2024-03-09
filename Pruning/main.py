import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from CGA.CGA import complete_gradient_algorithm
from CGA.calaculate_h import calculate_list_of_h
from CGA.cluster_CGA import cluster_algorithm
from CGA.kernel_density_estimator import gradient, kernel_density_estimator


def main():
    iris = load_iris()
    data = iris.data
    # data = np.array([[1.1, ], [0.1], [2.1], [1.12], [1.7], [15.0]])
    data = np.array([
    [3, 6],
    [3.3, 7],
    [5, 4],
    [31.3, 7],
    [61.3, 7],
])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, h = complete_gradient_algorithm(data)

    x = scaler.fit_transform(x)

    cluster_algorithm(x)


if __name__ == "__main__":
    main()