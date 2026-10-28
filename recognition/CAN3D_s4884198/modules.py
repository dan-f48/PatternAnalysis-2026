import torch.nn as nn 
import torch 
import torch.nn.functional as F


class Encoder(nn.Module):
    #(B, 1, H, W) -> (B, latent_dim, H/8, W/8)
    def __init__(self, latent_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            #compresses the image by half, generating 64 layers, then 2*64 etc. Last step converts 256 layers back to 64 (or the passed latent dim)
            nn.Conv2d(1, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden * 2, hidden * 4, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden * 4, latent_dim, 1), 
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    #(B, latent_dim, H/8, W/8) -> (B, 1, H, W), values in [0, 1]
    def __init__(self, latent_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            #reverses the encoder process 
            nn.Conv2d(latent_dim, hidden * 4, 1), nn.ReLU(),
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden, 1, 4, stride=2, padding=1), nn.Sigmoid(), #values in [0,1] 
        )

    def forward(self, x):
        return self.net(x)

    
class BaseModel(nn.Module):
    def __init__(self, latent_dim=64, hidden=64):
        super().__init__()
        self.encoder = Encoder(latent_dim, hidden)
        self.decoder = Decoder(latent_dim, hidden)

    def forward(self, x):
        return self.decoder(self.encoder(x)), 0.0

class BaseVAE(nn.Module):
    def __init__(self, latent_dim=64, hidden=64, beta=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(2* latent_dim, hidden) #outputs mu and logvar 
        self.decoder = Decoder(latent_dim, hidden)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1) #split channels in half 
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5*logvar) #reparametisaiton 
        else:
            z = mu 
        kl = -0.5*torch.mean(1+logvar - mu.pow(2) - logvar.exp())
        return self.decoder(z), self.beta * kl

    def sample(self, n, h=32, w=16, device="cpu"):
        z = torch.randn(n, self.latent_dim, h, w, device=device)
        return self.decoder(z)

    

class VectorQuantizerEMA(nn.Module):
    """Snaps each latent vector to its nearest codebook entry.
    Codebook is updated by exponential moving averages (no codebook loss term),
    with dead-code reset to avoid codebook collapse."""
 
    def __init__(self, num_codes=512, dim=64, decay=0.99, eps=1e-5,
                 commitment=0.25, dead_threshold=0.5):
        super().__init__()
        self.num_codes, self.dim = num_codes, dim
        self.decay, self.eps = decay, eps
        self.commitment, self.dead_threshold = commitment, dead_threshold
 
        embed = torch.randn(num_codes, dim)
        self.register_buffer("embed", embed)                       # the codebook
        self.register_buffer("cluster_size", torch.ones(num_codes))  # EMA of usage
        self.register_buffer("embed_avg", embed.clone())           # EMA of assigned vectors
 
    def forward(self, z):
        # z: (B, D, H, W) -> flat vectors (B*H*W, D)
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)
 
        # nearest codebook entry for every vector (squared L2 distance)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1)[None, :])
        idx = dist.argmin(dim=1)
        onehot = F.one_hot(idx, self.num_codes).type(flat.dtype)
        zq = F.embedding(idx, self.embed).view(B, H, W, D).permute(0, 3, 1, 2)
 
        if self.training:
            with torch.no_grad():
                # EMA update of usage counts and code sums
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                size = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
                self.embed.copy_(self.embed_avg / size.unsqueeze(1))
 
                # dead-code reset: re-seed unused codes with random encoder outputs
                dead = self.cluster_size < self.dead_threshold
                n_dead = int(dead.sum())
                if n_dead > 0:
                    pick = flat[torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)]
                    self.embed[dead] = pick
                    self.embed_avg[dead] = pick
                    self.cluster_size[dead] = 1.0
 
        # commitment loss pulls encoder outputs toward their chosen codes
        loss = self.commitment * F.mse_loss(z, zq.detach())
        zq = z + (zq - z).detach()  # straight-through gradient estimator
 
        avg = onehot.mean(0)
        perplexity = torch.exp(-(avg * torch.log(avg + 1e-10)).sum())  # ~ number of codes in use
        return zq, loss, idx.view(B, H, W), perplexity