
import os
import cv2
import numpy as np
from PIL import Image
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.inference import photo_to_wafer_array

def debug_cleaning():
    # Use one of the sample images
    sample_path = os.path.join(ROOT_DIR, "frontend", "samples", "none.png")
    if not os.path.exists(sample_path):
        print(f"Sample not found: {sample_path}")
        return

    pil = Image.open(sample_path)
    # The sample is already a map, but let's see what our photo-cleaning does to it
    arr = photo_to_wafer_array(pil)
    
    # Save the result visually
    mapping = np.array([0, 127, 255], dtype=np.uint8)
    img_uint8 = mapping[arr.astype(np.int32).clip(0, 2)]
    cleaned_pil = Image.fromarray(img_uint8, mode="L")
    
    debug_out = os.path.join(ROOT_DIR, "debug_cleaned_sample.png")
    cleaned_pil.save(debug_out)
    print(f"Debug image saved to {debug_out}")
    print(f"Array unique values: {np.unique(arr)}")
    print(f"Defect pixel count (val=2): {np.sum(arr == 2)}")
    print(f"Pass pixel count (val=1): {np.sum(arr == 1)}")

if __name__ == "__main__":
    debug_cleaning()
