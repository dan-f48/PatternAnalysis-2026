import time
import torch 
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure 
import matplotlib.pyplot as plt 

from dataset import get_dataloaders
from modules import BaseModel, BaseVAE

ROOT = "C:/Users/danfu/PatternAnalysis-2026/data/"
EPOCHS = 2
BATCH_SIZE = 64
SUBSET = None 
NUM_WORKERS = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #standard device def 

def main(model):
    train_loader, validate_loader, test_loader = get_dataloaders(ROOT, batch_size=BATCH_SIZE, subset=SUBSET, num_workers=NUM_WORKERS) #grab data using dataset.py
    #send given model, ssim to device and define adam optimiser
    model = model.to(device) 
    ckpt_path = "vae_best.pth"
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-1)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    #reset the tracked memory stats so measurement is accurate 
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_ssim = 0.0

    #training loop. One pass over the data per epoch 
    for epoch in range (EPOCHS):
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
        if val_ssim > best_ssim:
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
    model = BaseModel(latent_dim=64)
    main(model=model)

def train_vae():
    model = BaseVAE(latent_dim=64)
    main(model=model)
    save_samples(model, device, path="vae_sample.png")

if __name__ == "__main__":
    train_vae()