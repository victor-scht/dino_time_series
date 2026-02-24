import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def generate_dag(num_nodes=5, edge_prob=0.3):
    G = nx.DiGraph()

    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # garantit acyclic
            if np.random.rand() < edge_prob:
                G.add_edge(i, j)

    return G


def rbf_kernel(t, length_scale=10.0):
    t = t[:, None]
    return np.exp(-((t - t.T) ** 2) / (2 * length_scale**2))


def sample_gp(T):
    t = np.arange(T)

    # moyenne non stationnaire
    mu = np.sin(t / 10) + 0.1 * t / T

    K = rbf_kernel(t) + 1e-6 * np.eye(T)

    return np.random.multivariate_normal(mu, K)


def random_nonlinearity():
    funcs = [
        np.tanh,
        np.sin,
        lambda x: np.maximum(0, x),  # ReLU
        lambda x: x**2,
    ]
    return np.random.choice(funcs)


def generate_time_series_from_dag(G, T=512):
    signals = {}

    # ordre topologique
    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))

        if len(parents) == 0:
            # racine → GP
            signals[node] = sample_gp(T)
        else:
            # combinaison parents
            combined = np.zeros(T)

            for p in parents:
                w = np.random.normal()
                combined += w * signals[p]

            b = np.random.normal()
            f = random_nonlinearity()

            signals[node] = f(combined + b)

    return signals


def select_observed(signals, num_observed=3):
    keys = list(signals.keys())
    selected = np.random.choice(keys, num_observed, replace=False)

    return np.stack([signals[k] for k in selected], axis=0)


G = generate_dag(num_nodes=8)
signals = generate_time_series_from_dag(G, T=512)
x = select_observed(signals, num_observed=3)

print(x.shape)  # (channels, time)
plt.plot(x[2], c="blue")
plt.show()
