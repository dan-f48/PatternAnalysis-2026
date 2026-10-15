import torch.nn as nn 

class Encoder(nn.Module):
    #(B, 1, H, W) -> (B, latent_dim, H/8, W/8)
    def __init__(self, latent_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            #compresses the image by half, generating 64 layers, then 2*64 etc. Last step converts 256 layers back to 64 (or the passed latent dim)
            nn.Conv2d(1, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden * 2, hidden * 4, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden * 4, latent_dim, 1), nn.ReLU(),
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
            nn.Conv2d(hidden * 4, hidden * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden * 2, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    
class BaseModel(nn.Module):
    def __init__(self, latent_dim=64, hidden=64):
        super().__init__()
        self.encoder = Encoder(latent_dim, hidden)
        self.decoder = Decoder(latent_dim, hidden)

    def forward(self, x):
        return self.decoder(self.encoder(x))