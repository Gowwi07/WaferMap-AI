from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


CLASSES = ["normal", "center", "edge_ring", "scratch", "random_defect"]


def _normalize(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0)


def _make_wafer_mask(size: int) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    c = (size - 1) / 2
    r = size * 0.45
    return ((x - c) ** 2 + (y - c) ** 2) <= (r**2)


def _base(size: int) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    mask = _make_wafer_mask(size)
    img[mask] = 0.25
    noise = np.random.normal(0, 0.02, size=(size, size)).astype(np.float32)
    img = img + noise * mask
    return _normalize(img)


def _center(size: int) -> np.ndarray:
    img = _base(size)
    c = size // 2
    rr = size * 0.14
    y, x = np.ogrid[:size, :size]
    blob = ((x - c) ** 2 + (y - c) ** 2) <= (rr**2)
    img[blob] = 0.95
    return _normalize(img)


def _edge_ring(size: int) -> np.ndarray:
    img = _base(size)
    y, x = np.ogrid[:size, :size]
    c = (size - 1) / 2
    dist = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    r_outer = size * 0.44
    r_inner = size * 0.38
    ring = (dist <= r_outer) & (dist >= r_inner)
    img[ring] = 0.9
    return _normalize(img)


def _scratch(size: int) -> np.ndarray:
    img = _base(size)
    x0 = random.randint(size // 6, size // 3)
    y0 = random.randint(size // 6, size // 3)
    angle = random.uniform(-math.pi / 4, math.pi / 4)
    length = random.randint(size // 2, int(size * 0.8))
    for i in range(length):
        x = int(x0 + i * math.cos(angle))
        y = int(y0 + i * math.sin(angle))
        if 0 <= x < size and 0 <= y < size:
            img[max(0, y - 1) : min(size, y + 2), max(0, x - 1) : min(size, x + 2)] = 0.95
    return _normalize(img)


def _random_defect(size: int) -> np.ndarray:
    img = _base(size)
    n = random.randint(12, 40)
    for _ in range(n):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        rr = random.randint(1, 3)
        img[max(0, y - rr) : min(size, y + rr + 1), max(0, x - rr) : min(size, x + rr + 1)] = 0.9
    return _normalize(img)


GENS: dict[str, Callable[[int], np.ndarray]] = {
    "normal": _base,
    "center": _center,
    "edge_ring": _edge_ring,
    "scratch": _scratch,
    "random_defect": _random_defect,
}


def write_dataset(out_dir: Path, n_per_class: int = 500, size: int = 128) -> None:
    for cls in CLASSES:
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        gen = GENS[cls]
        for i in range(n_per_class):
            img = (gen(size) * 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(cls_dir / f"{cls}_{i:04d}.png")


if __name__ == "__main__":
    root = Path("data/wafer_synth")
    write_dataset(root, n_per_class=400, size=128)
    print(f"Synthetic dataset generated at {root.resolve()}")
