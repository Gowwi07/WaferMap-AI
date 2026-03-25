from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_dataloaders(
    data_root: str | Path,
    img_size: int = 128,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 0,
):
    data_root = Path(data_root)
    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    full_ds = datasets.ImageFolder(root=str(data_root), transform=tfm)
    n_train = int(len(full_ds) * train_split)
    n_val = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    class_names = full_ds.classes
    return train_loader, val_loader, class_names
