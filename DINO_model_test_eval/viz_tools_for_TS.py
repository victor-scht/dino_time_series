import torch
import numpy as np
import matplotlib.pyplot as plt
from time_series_generation import *
# [Assuming the functions rbf_kernel, periodic_kernel, sample_gp_like, 
# random_nonlinearity, and SyntheticDAGDataset are defined above]

def visualize_diversity(n_plots=4):
    # Setup dataset
    # T=200 gives us a good look at the periodic patterns
    dataset = SyntheticDAGDataset(n_samples=n_plots, T=200, n_nodes=10, n_roots=3, n_obs=3)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    for i in range(n_plots):
        # Get one sample: shape (n_obs, T)
        sample = dataset[i].numpy()
        
        ax = axes[i]
        for obs_idx in range(sample.shape[0]):
            ax.plot(sample[obs_idx], label=f"Observed Node {obs_idx}", alpha=0.8)
        
        ax.set_title(f"Sample {i+1}: Complex Causal Interactions")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(alpha=0.3)

    plt.xlabel("Time (T)")
    plt.show()
    
def plot_ecg_samples(X, y, n_examples=3):
    # On crée une figure avec deux colonnes : Classe 0 et Classe 1
    classes = np.unique(y)
    fig, axes = plt.subplots(n_examples, len(classes), figsize=(12, 4 * n_examples), sharey=True)

    for col, cls in enumerate(classes):
        # On récupère les indices des échantillons de cette classe
        idx_cls = np.where(y == cls)[0]
        
        for row in range(n_examples):
            # Sélection aléatoire d'un ECG de cette classe
            idx = np.random.choice(idx_cls)
            signal = X[idx, 0, :]  # Shape (T,) car on prend le canal 0
            
            ax = axes[row, col]
            ax.plot(signal, color='red' if cls == 1 else 'blue', alpha=0.8)
            ax.set_title(f"Classe {cls} - Échantillon #{idx}")
            ax.grid(True, alpha=0.3)
            if row == n_examples - 1:
                ax.set_xlabel("Points (T=256)")

    plt.tight_layout()
    plt.show()

