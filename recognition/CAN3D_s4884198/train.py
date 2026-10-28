import time
import torch 
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure 
import matplotlib.pyplot as plt 

from dataset import get_dataloaders
from modules import BaseModel, BaseVAE, VQVAE, CodePrior

ROOT = "C:/Users/danfu/PatternAnalysis-2026/data/"
EPOCHS = 80
BATCH_SIZE = 64
SUBSET = None 
NUM_WORKERS = 0
TARGET_BETA = 1e-1
WARMUP = 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #standard device def 

def main(model, ckpt_path):
    train_loader, validate_loader, test_loader = get_dataloaders(ROOT, batch_size=BATCH_SIZE, subset=SUBSET, num_workers=NUM_WORKERS) #grab data using dataset.py
    #send given model, ssim to device and define adam optimiser
    model = model.to(device) 
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    #reset the tracked memory stats so measurement is accurate 
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_ssim = 0.0

    #training loop. One pass over the data per epoch 
    for epoch in range (EPOCHS):
        model.beta = TARGET_BETA * min(1.0, epoch / 15)
        start = time.time()
        model.train() #model in training mode 
        total = 0.0 
        #loads a batch from the dataloader (subset of 32 shuffled training images)
        for x in train_loader:
            x = x.to(device)
            recon, extra_loss = model(x)
            loss = F.mse_loss(recon, x) + extra_loss #MSE of original from model reconstruction 
            optimiser.zero_grad() #clear old gradients 
            #backpropogate 
            loss.backward() 
            optimiser.step()

            #adds loss weighted by batch size (so the smaller batch at the end doesn't skew data)
            total += loss.item() * len(x)
        train_loss = total / len(train_loader.dataset)        
        model.eval() #eval mode
        ssim.reset()

        #model validation after each epoch is done 
        with torch.no_grad():
            #loads batches from validation loader and generates SSIM 
            for x in validate_loader:
                x = x.to(device)
                recon, _ = model(x)
                ssim.update(recon, x)
        val_ssim = ssim.compute().item()
        if epoch == 0 or (epoch >= WARMUP and val_ssim > best_ssim):
            best_ssim = val_ssim
            torch.save(model.state_dict(), ckpt_path)


        print(f"epoch: {epoch + 1}: train MSE {train_loss:.4f}, validate SSIM {ssim.compute():.3f}, {time.time() - start:.1f} seconds")

    ssim.reset()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    #model testing after the training is done 
    with torch.no_grad():
        #loads batches from test loader and generates SSIM 
        for x  in test_loader:
            x = x.to(device)
            recon, _ = model(x)
            ssim.update(recon, x) 
    print(f"test SSIM: {ssim.compute():.3f}")

    #stats if i ran it on RANGPUR
    if device.type == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def save_samples(model, device, path="samples.png", n=6, seed=0):
    model.eval()
    torch.manual_seed(seed)

    with torch.no_grad():
        images = model.sample(n, device=device).cpu()

    fig, axes = plt.subplots(1, n, figsize=(2*n, 4))
    for ax, img in zip(axes, images):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle("generated samples")
    plt.savefig(path)
    plt.close()
    
def train_basemodel():
    main(BaseModel(latent_dim=64), "base_best.pth")

def train_vae():
    model = BaseVAE(latent_dim=8)
    main(model, "vae_best.pth")
    save_samples(model, device, path="vae_sample.png")

def train_vq_vae():
    main(VQVAE(), VQ_CKPT)

VQ_CKPT = "vqvae_best.pth"
PRIOR_CKPT = "prior_best.pth"
PRIOR_EPOCHS = 30
PRIOR_BATCH_SIZE = 64
PRIOR_N_SAMPLES = 16
SEED = 0

@torch.no_grad()
def encode_split(vq, loader):
    """Turn every image in a split into its (H*W,) sequence of code indices."""
    out = []
    for x in loader:
        out.append(vq.encode_indices(x.to(device)).flatten(1).cpu())
    return torch.cat(out)
 
 
def save_grid(imgs, path, title):
    cols = 8
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img in zip(axes.ravel(), imgs):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.suptitle(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
 
 
def train_prior():
    torch.manual_seed(SEED)
    train_loader, val_loader, _ = get_dataloaders(ROOT, batch_size=PRIOR_BATCH_SIZE)
 
    vq = VQVAE().to(device)
    vq.load_state_dict(torch.load(VQ_CKPT, map_location=device))
    vq.eval()
 
    # encode once; the prior trains on code sequences, not images
    train_codes, val_codes = encode_split(vq, train_loader), encode_split(vq, val_loader)
    seq_len = train_codes.shape[1]
    sample_x = next(iter(val_loader)).to(device)
    h, w = vq.encode_indices(sample_x[:1]).shape[1:]  # latent grid size (e.g. 32 x 16)
    print(f"codes: train {tuple(train_codes.shape)}, val {tuple(val_codes.shape)}, grid {h}x{w}")
 
    prior = CodePrior(num_codes=vq.quant.num_codes, seq_len=seq_len).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=3e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
 
    best = float("inf")
    for epoch in range(PRIOR_EPOCHS):
        start = time.time()
        prior.train()
        perm = torch.randperm(len(train_codes))
        total = 0.0
        for i in range(0, len(perm), PRIOR_BATCH_SIZE):
            codes = train_codes[perm[i:i + PRIOR_BATCH_SIZE]].to(device)
            loss = prior.loss(codes)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)  # gradient hygiene
            opt.step()
            total += loss.item() * len(codes)
 
        prior.eval()
        with torch.no_grad():
            val = sum(prior.loss(val_codes[i:i + PRIOR_BATCH_SIZE].to(device)).item() * len(val_codes[i:i + PRIOR_BATCH_SIZE])
                      for i in range(0, len(val_codes), PRIOR_BATCH_SIZE)) / len(val_codes)
        if val < best:
            best = val
            torch.save(prior.state_dict(), PRIOR_CKPT)
        print(f"epoch {epoch + 1}: train CE {total / len(train_codes):.3f}, "
              f"val CE {val:.3f}, {time.time() - start:.1f}s")
 
    if device.type == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
 
    # generate with fixed seed (uncurated)
    prior.load_state_dict(torch.load(PRIOR_CKPT, map_location=device))
    prior.eval()
    torch.manual_seed(SEED)
    t0 = time.time()
    codes = prior.sample(PRIOR_N_SAMPLES, device=device).view(PRIOR_N_SAMPLES, h, w)
    imgs = vq.decode_indices(codes).cpu()
    print(f"generation time for {PRIOR_N_SAMPLES} images: {time.time() - t0:.1f}s")
    save_grid(imgs, "vqvae_generated.png", f"VQ-VAE + prior samples (seed {SEED})")
    print("saved vqvae_generated.png")

#when training vae set the warmup to 15

if __name__ == "__main__":
    train_vq_vae()
    train_prior()