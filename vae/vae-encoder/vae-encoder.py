import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    x: [bs, dim]
    """
    # Your implementation here
    def _relu(input):
        return np.maximum(0, input)

    input_dim = x.shape[1]
    
    # linear layer 1: [bs, dim] @ [dim, dim1] + [1, dim1] -> [bs, dim1]
    w1 = np.random.rand(input_dim, input_dim)
    b1 = np.random.rand(1, input_dim)
    
    # linear layer 2: [bs, dim1] -> [bs, latent_dim]
    w2_mean = np.random.rand(input_dim, latent_dim)
    b2_mean = np.random.rand(1, latent_dim)
    w2_logvar = np.random.rand(input_dim, latent_dim)
    b2_logvar = np.random.rand(1, latent_dim)

    # forward
    a1 = x @ w1 + b1
    a1 = _relu(a1)
    mean = a1 @ w2_mean + b2_mean
    log_var = a1 @ w2_logvar + b2_logvar
    
    return (mean, log_var)