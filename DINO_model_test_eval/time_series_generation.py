import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tslearn.datasets import UCR_UEA_datasets


def rbf_kernel(t, lengthscale=30.0, var=1.0):
    d = (t[:, None] - t[None, :]) ** 2
    return var * np.exp(-0.5 * d / (lengthscale**2 + 1e-9))


def periodic_kernel(t, period=50.0, lengthscale=10.0, var=0.5):
    d = np.abs(t[:, None] - t[None, :])
    return var * np.exp(
        -2.0 * (np.sin(np.pi * d / (period + 1e-9)) ** 2) / (lengthscale**2 + 1e-9)
    )


def sample_gp_like(T: int, rng: np.random.RandomState):
    t = np.arange(T).astype(np.float32)

    # non-stationary mean: trend + season
    a = rng.uniform(-0.02, 0.02)
    b = rng.uniform(-1.0, 1.0)
    season_amp = rng.uniform(0.0, 1.0)
    season_per = rng.uniform(20.0, 120.0)
    mu = a * t + b + season_amp * np.sin(2 * np.pi * t / season_per)

    # random composed kernel: RBF + Periodic + noise
    ls = rng.uniform(10.0, 80.0)
    var = rng.uniform(0.3, 2.0)
    K = rbf_kernel(t, ls, var)

    if rng.rand() < 0.7:
        per = rng.uniform(20.0, 150.0)
        pls = rng.uniform(5.0, 40.0)
        pvar = rng.uniform(0.1, 1.0)
        K = K + periodic_kernel(t, per, pls, pvar)

    K = K + np.eye(T, dtype=np.float32) * rng.uniform(1e-3, 5e-2)

    # sample via cholesky
    L = np.linalg.cholesky(K + 1e-6 * np.eye(T, dtype=np.float32))
    z = rng.randn(T).astype(np.float32)
    x = mu + L @ z
    return x.astype(np.float32)


def random_nonlinearity(name: str):
    if name == "tanh":
        return np.tanh
    if name == "relu":
        return lambda x: np.maximum(0.0, x)
    if name == "sin":
        return np.sin
    if name == "cubic":
        return lambda x: x**3
    return np.tanh


class SyntheticDAGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        n_samples: int,
        T: int,
        n_nodes: int = 8,
        n_roots: int = 3,
        n_obs: int = 1,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.T = T
        self.n_nodes = n_nodes
        self.n_roots = n_roots
        self.n_obs = n_obs
        self.rng = np.random.RandomState(seed)

        # fixed random DAG: edges only from lower index to higher index => acyclic
        self.adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)
        for j in range(n_roots, n_nodes):
            # pick 1-3 parents among earlier nodes
            parents = self.rng.choice(
                np.arange(0, j), size=self.rng.randint(1, min(4, j + 1)), replace=False
            )
            self.adj[parents, j] = 1

        # per-node nonlinearities
        funcs = ["tanh", "relu", "sin", "cubic"]
        self.f = [random_nonlinearity(self.rng.choice(funcs)) for _ in range(n_nodes)]

        # observed nodes chosen from non-roots by default
        candidates = np.arange(n_roots, n_nodes)
        self.obs_nodes = self.rng.choice(candidates, size=n_obs, replace=False)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.rng.randint(0, 10_000_000))

        X = np.zeros((self.n_nodes, self.T), dtype=np.float32)

        # roots
        for v in range(self.n_roots):
            X[v] = sample_gp_like(self.T, rng)

        # non-roots
        for v in range(self.n_roots, self.n_nodes):
            parents = np.where(self.adj[:, v] == 1)[0]
            s = np.zeros(self.T, dtype=np.float32)
            for u in parents:
                w = rng.randn()  # N(0,1)
                s += w * X[u]
            b = rng.randn()
            X[v] = self.f[v](s + b).astype(np.float32)

        obs = X[self.obs_nodes]  # (C, T)
        # normalize lightly for stability (tokenizer does instance norm too)
        obs = (obs - obs.mean(axis=-1, keepdims=True)) / (
            obs.std(axis=-1, keepdims=True) + 1e-6
        )
        return torch.from_numpy(obs)  # (C, T)


#################################################################
# ECG prep
#################################################################
def resample_to_T(x_np: np.ndarray, T: int) -> np.ndarray:
    """
    Standardise et rééchantillonne un signal vers une longueur T.
    Entrée : (B, T) ou (B, T, C) ou (B, C, T)
    Sortie : Toujours (B, C, T)
    """
    # 1. Conversion initiale en tenseur
    x = torch.from_numpy(x_np).float()

    # 2. Gestion des dimensions (B, T) -> (B, 1, T)
    if x.ndim == 2:
        x = x.unsqueeze(1)

    # 3. Gestion des dimensions (B, T, C) -> (B, C, T)
    elif x.ndim == 3:
        # Heuristique tslearn : (Samples, Timesteps, Channels)
        # On permute si la dernière dim est petite (canaux)
        # OU si on sait que tslearn charge souvent du (B, T, C)
        if x.shape[2] <= 16 and x.shape[1] > x.shape[2]:
            x = x.permute(0, 2, 1)
        # Sinon on suppose déjà (B, C, T), aucune action requise
    else:
        raise ValueError(f"Shape inattendue: {x.shape}")

    # 4. Rééchantillonnage (Interpolation linéaire)
    # F.interpolate attend (Batch, Channel, Length)
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)

    return x.numpy()
