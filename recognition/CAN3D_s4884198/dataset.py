import glob, numpy as np, nibabel as nib, torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

IMG_DIRS = {
    "train": "train",
    "validate": "validate",
    "test": "test"
}

SEG_DIRS = {
    "train": "train",
    "validate": "validate",
    "test": "test"
}


def load_slices(paths, is_mask=False):
    arrs = []
    for p in tqdm(paths):
        a = nib.load(p).get_fdata(caching="unchanged") # loops through files reading them into np array of pixel values
        if a.ndim == 3:
            a = a[:, :, 0]                      # drop extra dimemsions in case images have a third dimension
        a = a.astype(np.float32) #np expected type
        if is_mask:
            a = (a > 0).astype(np.float32)      # binary prostate maskc (prostate or not)
        else:
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)   # [0, 1] (small value stops a divide by 0 err)
        arrs.append(a)
    return np.stack(arrs)[:, None]              # (N, 1, H, W)