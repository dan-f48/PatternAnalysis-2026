"""predict.py - check vae_best.pth reconstructs real test images (no latent sampling)."""
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; figures are saved, not shown
import matplotlib.pyplot as plt
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import _find, load_slices, pad_multiple, IMG_DIRS
from modules import BaseVAE

ROOT = "C:/Users/danfu/PatternAnalysis-2026/data/"
CKPT = "vae_best.pth"
N_SHOW = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(ckpt=CKPT, out="vae_predictions.png"):
    # load model
    model = BaseVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()  # uses the encoder mean, so reconstructions are deterministic

    # load the whole test split (small, loads in seconds)
    paths = _find(ROOT, IMG_DIRS["test"])
    x = pad_multiple(torch.from_numpy(load_slices(paths)), 8)

    # reconstruct in batches and score every slice
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)
    recons, scores = [], []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            batch = x[i:i + 64].to(device)
            recon, _ = model(batch)
            # score each slice individually
            scores.append(torch.stack([ssim(recon[j:j + 1], batch[j:j + 1]) for j in range(len(batch))]).cpu())
            recons.append(recon.cpu())
    recon = torch.cat(recons)
    scores = torch.cat(scores)
    print(f"test slices: {len(x)}")
    print(f"mean SSIM: {scores.mean():.3f}  min: {scores.min():.3f}  max: {scores.max():.3f}")

    # evenly spaced slices (not cherry-picked): original / reconstruction / abs error
    idx = torch.linspace(0, len(x) - 1, N_SHOW).long()
    fig, axes = plt.subplots(3, N_SHOW, figsize=(2 * N_SHOW, 7))
    for col, i in enumerate(idx):
        axes[0, col].imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(recon[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow((x[i] - recon[i]).abs().squeeze(), cmap="hot", vmin=0, vmax=0.5)
        axes[0, col].set_title(f"SSIM {scores[i]:.2f}", fontsize=9)
        for row in range(3):
            axes[row, col].axis("off")
    axes[0, 0].text(-0.1, 0.5, "Original", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.1, 0.5, "Recon", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    axes[2, 0].text(-0.1, 0.5, "|Error|", transform=axes[2, 0].transAxes, rotation=90, va="center", ha="right")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def generate(ckpt=CKPT, out="vae_generated.png", n=8, seed=0, match_data=False):
    """Decode random latents into new images.

    match_data=False: sample z ~ N(0, 1) (standard VAE sampling).
    match_data=True:  sample from a per-channel Gaussian fitted to encoded
                      training slices (fixes the prior/aggregate-posterior mismatch;
                      state this clearly if you report results from it).
    """
    model = BaseVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    torch.manual_seed(seed)  # fixed seed so the grid is reproducible, not cherry-picked

    h, w = 32, 16  # latent grid size for 256 x 128 inputs
    with torch.no_grad():
        if not match_data:
            z = torch.randn(n, model.latent_dim, h, w, device=device)
        else:
            paths = _find(ROOT, IMG_DIRS["train"])
            step = max(1, len(paths) // 500)  # ~500 evenly spaced training slices
            x = pad_multiple(torch.from_numpy(load_slices(paths[::step])), 8).to(device)
            mu, _ = model.encoder(x).chunk(2, dim=1)
            mean = mu.mean(dim=(0, 2, 3), keepdim=True)  # per-channel statistics
            std = mu.std(dim=(0, 2, 3), keepdim=True)
            z = mean + std * torch.randn(n, model.latent_dim, h, w, device=device)
        imgs = model.decoder(z).cpu()

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 4))
    for ax, img in zip(axes, imgs):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle(f"Generated samples (seed {seed}, {'fitted' if match_data else 'N(0,1)'} latent)")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    # python predict.py                      -> reconstruction check
    # python predict.py generate [match]     -> generate new images
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate(match_data=(len(sys.argv) > 2 and sys.argv[2] == "match"))
    else:
        main(*sys.argv[1:])