import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    # Your implementation here
    epsilon = np.random.standard_normal(mu.shape)
    var = np.exp(0.5 * log_var)
    z = mu + var * epsilon
    return z 