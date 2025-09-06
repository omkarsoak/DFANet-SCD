## Dataloader
from torch.utils.data import Dataset, DataLoader
import os
import rasterio
import torch
import numpy as np

class ChangeDetectionDatasetTIF(Dataset):
    def __init__(self, t2019_dir, t2024_dir, sem_2019_dir, sem_2024_dir, mask_dir,
                 classes, semantic_classes, transform=None):
        self.t2019_dir = t2019_dir
        self.t2024_dir = t2024_dir
        self.mask_dir = mask_dir
        self.sem_2019_dir = sem_2019_dir
        self.sem_2024_dir = sem_2024_dir
        self.classes = classes  # Change detection classes
        self.semantic_classes = semantic_classes  # Land cover classes
        self.transform = transform

        # Load all paths
        self.t2019_paths = sorted([f for f in os.listdir(t2019_dir) if f.endswith('.tif')])
        self.t2024_paths = sorted([f for f in os.listdir(t2024_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        self.sem2019_paths = sorted([f for f in os.listdir(sem_2019_dir) if f.endswith('.tif')])
        self.sem2024_paths = sorted([f for f in os.listdir(sem_2024_dir) if f.endswith('.tif')])

        # Verify all paths match
        assert len(self.t2019_paths) == len(self.t2024_paths) == len(self.mask_paths) == \
               len(self.sem2019_paths) == len(self.sem2024_paths), "Mismatched number of images"

    def __len__(self):
        return len(self.t2019_paths)

    def __getitem__(self, index):
        # Load images using rasterio
        with rasterio.open(os.path.join(self.t2019_dir, self.t2019_paths[index])) as src:
            img_t2019 = src.read(out_dtype=np.float32) / 255.0
        with rasterio.open(os.path.join(self.t2024_dir, self.t2024_paths[index])) as src:
            img_t2024 = src.read(out_dtype=np.float32) / 255.0

        # Load masks
        with rasterio.open(os.path.join(self.mask_dir, self.mask_paths[index])) as src:
            cd_mask = src.read(1).astype(np.int64)
        with rasterio.open(os.path.join(self.sem_2019_dir, self.sem2019_paths[index])) as src:
            sem_mask_2019 = src.read(1).astype(np.int64)
        with rasterio.open(os.path.join(self.sem_2024_dir, self.sem2024_paths[index])) as src:
            sem_mask_2024 = src.read(1).astype(np.int64)

        # Convert to PyTorch tensors
        img_t2019 = torch.from_numpy(img_t2019)
        img_t2024 = torch.from_numpy(img_t2024)
        cd_mask = torch.from_numpy(cd_mask)
        sem_mask_2019 = torch.from_numpy(sem_mask_2019)
        sem_mask_2024 = torch.from_numpy(sem_mask_2024)

        # Apply transforms if any
        if self.transform is not None:
            img_t2019 = self.transform(img_t2019)
            img_t2024 = self.transform(img_t2024)

        # Return dictionary format
        return {
            'img_t2019': img_t2019,
            'img_t2024': img_t2024,
            'cd_mask': cd_mask,
            'sem_mask_2019': sem_mask_2019,
            'sem_mask_2024': sem_mask_2024
        }

def describe_loader(loader_type):
    """Print information about a data loader"""
    sample = next(iter(loader_type))
    print("Batch size:", loader_type.batch_size)
    print("Shapes:")
    print("  Image 2019:", sample['img_t2019'].shape)
    print("  Image 2024:", sample['img_t2024'].shape)
    print("  Change Mask:", sample['cd_mask'].shape)
    print("  Semantic Mask 2019:", sample['sem_mask_2019'].shape)
    print("  Semantic Mask 2024:", sample['sem_mask_2024'].shape)
    print("Number of images:", len(loader_type.dataset))
    print("Change Classes:", loader_type.dataset.classes)
    print("Semantic Classes:", loader_type.dataset.semantic_classes)

    # Print value ranges
    print("\nValue ranges:")
    #print("  Images:", torch.min(sample['img_t2019']).item(), "to", torch.max(sample['img_t2019']).item())
    print("  Change Mask:", torch.unique(sample['cd_mask']))
    print("  Semantic Mask:", torch.unique(sample['sem_mask_2019']))
