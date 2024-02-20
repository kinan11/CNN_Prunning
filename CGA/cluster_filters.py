import numpy as np
from sklearn.preprocessing import StandardScaler
from CGA.CGA import complete_gradient_algorithm
from CGA.cluster_CGA import cluster_algorithm


def cluster_filters(filters):
    scaler = StandardScaler()
    filters_flat = np.array([f.reshape(-1) for f in filters])
    scaler = StandardScaler()
    filters_scaled = scaler.fit_transform(filters_flat)

    x, h = complete_gradient_algorithm(filters_scaled)
    return [1, 2]
    # return cluster_algorithm(x, h)
