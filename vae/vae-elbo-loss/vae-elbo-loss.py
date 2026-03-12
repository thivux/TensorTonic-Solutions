import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """
    # Your implementation here
    # MSE: sum over features, mean over batch
    sqrt_matrix = np.pow(x - x_recon, 2)
    sum_over_features = np.sum(sqrt_matrix, 1)
    recon = np.mean(sum_over_features)

    # MSE: sum over features & batch
    kl = -1/2 * np.sum(1 + log_var - np.pow(mu, 2) - np.exp(log_var))
    
    total = recon + kl
    return {"total":  total, "recon": recon, "kl": kl}