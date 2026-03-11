import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    # Your implementation here
    def _sigmoid(x): 
        return 1 / (1 + np.exp(-x))
        
    latent_dim = z.shape[1]

    # layer 1: [bs, latent_dim] @ [latent_dim, output_dim] + [1, output_dim] -> [bs, output_dim]
    w1 = np.random.rand(latent_dim, output_dim)
    b1 = np.random.rand(1, output_dim)
    a1 = _sigmoid(z @ w1 + b1) 

    # layer 2: [bs, output_dim] @ [output_dim, output_dim] + [1, output_dim] -> [bs, output_dim]
    w2 = np.random.rand(output_dim, output_dim)
    b2 = np.random.rand(1, output_dim)
    x_hat = a1 @ w2 + b2
    
    return x_hat