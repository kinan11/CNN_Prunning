import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from CGA.CGA import complete_gradient_algorithm
from CGA.cluster_CGA import cluster_algorithm


def main():
    # mean1 = [-1, 0, -1, -1]
    # mean2 = [1, 0, 1, 1]
    # mean3 = [0, -1, 0, 0]
    #
    # cluster1 = [np.random.normal(mean1) + [0, 0, 0, 0] for _ in range(200)]
    # cluster2 = [np.random.normal(mean2) + [20, 20, 20, 20] for _ in range(200)]
    # cluster3 = [np.random.normal(mean3) + [40, 40, 40, 40] for _ in range(200)]
    # data = np.concatenate((cluster1, cluster2, cluster3), axis=0)

    mean1 = [-1]
    mean2 = [1]
    mean3 = [0]

    cluster1 = [np.random.normal(mean1) + [0] for _ in range(10)]
    cluster2 = [np.random.normal(mean2) + [20] for _ in range(10)]
    cluster3 = [np.random.normal(mean3) + [-40] for _ in range(10)]
    data = np.concatenate((cluster1, cluster2, cluster3), axis=0)

    # data = np.concatenate((cluster1, cluster2), axis=0)

    # np.random.shuffle(data)

    # data = np.array([[1.], [2.], [3.]])

    # iris = load_iris()
    # data = iris.data
#     # data = np.array([[1.1, ], [0.1], [2.1], [1.12], [1.7], [15.0]])
#     data = np.array([
#     [3, 6],
#     [3.3, 7],
#     [5, 4],
#     [31.3, 7],
#     [61.3, 7],
# ])
    scaler = StandardScaler()

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, projection='3d')
    # ax2.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o')
    # plt.title('Oryginał')
    # plt.show()

    fig, ax = plt.subplots()
    ax.scatter(data, np.zeros_like(data), marker='o')
    plt.title('Oryginał')
    plt.show()

    # data = scaler.fit_transform(data)
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
            ax.scatter(x[index, 0],0, c=colors[i], marker='o')
            # ax.scatter(x[index, 0], x[index, 1], x[index, 2],  c=colors[i%6], marker='o')
        i += 1
    plt.title("Po CGCA")
    plt.show()


if __name__ == "__main__":
    main()
    # gradient(np.array([[1, 1], [2, 2], [3, 3]]), [1, 2], [1, 1, 1])