
import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_WM811K = os.path.join(ROOT_DIR, "samples_wm811k")
OUT_MIXED38 = os.path.join(ROOT_DIR, "samples_mixed38")

# 811K labels
WM811K_LABELS = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-Full', 'Random', 'Scratch', 'None']
# Mixed labels
MIXED_LABELS = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-Full', 'Random', 'Scratch']

def extract_wm811k():
    print("[Extract] Processing WM-811K...")
    pkl_path = os.path.join(ROOT_DIR, "data", "LSWMD.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found")
        return

    df = pd.read_pickle(pkl_path)
    # The 'failureType' is the label, 'waferMap' is the array
    # Labels are 1D arrays or strings sometimes, sanitize it.
    
    samples_per_class = 4
    for label in WM811K_LABELS:
        # Filter for this class
        # (Assuming failureType is formatted as ['Center'] or 'Center')
        subset = df[df['failureType'].apply(lambda x: x[0][0] if isinstance(x, np.ndarray) and len(x)>0 and len(x[0])>0 else str(x)) == label]
        if subset.empty:
            continue
            
        pick = subset.sample(min(samples_per_class, len(subset)))
        for i, row in pick.iterrows():
            arr = row['waferMap']
            # Map values 0,1,2 to 0,127,255
            mapping = np.array([0, 127, 255], dtype=np.uint8)
            img_uint8 = mapping[arr.astype(np.int32).clip(0, 2)]
            im = Image.fromarray(img_uint8, mode="L").resize((224, 224), Image.NEAREST)
            fname = f"wm811k_{label}_{i}.png"
            im.save(os.path.join(OUT_WM811K, fname))
    print(f"Done! {len(os.listdir(OUT_WM811K))} images in {OUT_WM811K}")

def extract_mixed38():
    print("[Extract] Processing MixedWM38...")
    npz_path = os.path.join(ROOT_DIR, "data", "MixedWM38.npz")
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found")
        return

    data = np.load(npz_path)
    arrays = data['arr_0']
    labels = data['arr_1']
    
    # Select 30 random samples
    indices = random.sample(range(len(arrays)), 30)
    for idx in indices:
        arr = arrays[idx]
        lbl = labels[idx]
        # Find which labels are 1
        active_labels = [MIXED_LABELS[i] for i, v in enumerate(lbl) if v == 1]
        label_str = "_".join(active_labels) if active_labels else "None"
        
        mapping = np.array([0, 127, 255], dtype=np.uint8)
        img_uint8 = mapping[arr.astype(np.int32).clip(0, 2)]
        im = Image.fromarray(img_uint8, mode="L").resize((224, 224), Image.NEAREST)
        fname = f"mixed_{label_str}_{idx}.png"
        im.save(os.path.join(OUT_MIXED38, fname))
    print(f"Done! {len(os.listdir(OUT_MIXED38))} images in {OUT_MIXED38}")

if __name__ == "__main__":
    extract_wm811k()
    extract_mixed38()
