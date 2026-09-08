"""
This file was created to provide the data
loaders, datasets, and variables necessary
for handling data in this project.
"""

# ============imports==============
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path("../data")
SEED = 42


# ============functions==============
def provider(data_path):
    """Return Loaders ; Datasets ; Variables"""
    # ============Base_transformer==============
    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # single channel images or gray scale.
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # ============load_dataset==============
    train_full = datasets.FashionMNIST(
        root=data_path, train=True, download=True, transform=base_transform
    )
    test_full = datasets.FashionMNIST(
        root=data_path, train=False, download=True, transform=base_transform
    )

    train_limit = 4000
    test_limit = 1000

    train_dataset = Subset(train_full, range(train_limit))
    test_dataset = Subset(test_full, range(test_limit))

    # ============Data_Loaders==============
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    class_names = train_full.classes
    return train_loader, test_loader, class_names


# ============test_block==============
if __name__ == "__main__":
    train_batch_loader, _, classes_names = provider(data_path=data_root)
    images, labels = next(iter(train_batch_loader))
    print(f"Gray Scaled image shape: {images.shape}")
    print(f"Class names: \n{classes_names}")
