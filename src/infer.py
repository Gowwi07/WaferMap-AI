from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import WaferNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="artifacts/wafernet.pt")
    parser.add_argument("--labels", default="artifacts/labels.json")
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    model = WaferNet(num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    img = Image.open(args.image)
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        idx = int(torch.argmax(probs).item())

    print(f"Predicted class: {labels[idx]}")
    print(f"Confidence: {float(probs[idx]):.4f}")


if __name__ == "__main__":
    main()
