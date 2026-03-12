import numpy as np

class VAE:
    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Initialize weights here
        # encoder
        self.w1_enc = np.random.rand(input_dim, latent_dim)
        self.b1_enc = np.random.rand(1, latent_dim)
        
        self.w2_enc_mu = np.random.rand(latent_dim, latent_dim)
        self.b2_enc_mu = np.random.rand(1, latent_dim)
        
        self.w2_enc_log_var = np.random.rand(latent_dim, latent_dim)
        self.b2_enc_log_var = np.random.rand(1, latent_dim)

        # decoder
        self.w1_dec = np.random.rand(latent_dim, input_dim)
        self.b1_dec = np.random.rand(1, input_dim)
        
        self.w2_dec = np.random.rand(input_dim, input_dim)
        self.b2_dec = np.random.rand(1, input_dim)
    
    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through VAE.
        """
        # Your implementation here
        # encode
        a1 = self._relu(x @ self.w1_enc + self.b1_enc) # bs, latent_dim
        mu = a1 @ self.w2_enc_mu + self.b2_enc_mu # bs, latent_dim
        log_var = a1 @ self.w2_enc_log_var + self.b2_enc_log_var # bs, latent_dim

        # reparam trick 
        epsilon = np.random.standard_normal(mu.shape)
        z_std = np.exp(0.5 * log_var)
        z = mu + epsilon * z_std # bs, latent_dim
        
        # decode
        a1_dec = self._sigmoid(z @ self.w1_dec + self.b1_dec) # bs, input_dim 
        x_hat = a1_dec @ self.w2_dec + self.b2_dec

        return (x_hat, mu, log_var)
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples from prior.
        """
        # Your implementation here
        # sample z
        z = np.random.standard_normal((n_samples, self.latent_dim))

        # decode
        a1_dec = self._sigmoid(z @ self.w1_dec + self.b1_dec) # bs, input_dim
        x_hat = self._sigmoid(a1_dec @ self.w2_dec + self.b2_dec) # bs, input_dim 
        
        return x_hat 