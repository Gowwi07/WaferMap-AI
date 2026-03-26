import os
from PIL import Image
from src.dataset import load_raw_dataframe, CLASS_NAMES, wafer_to_image

if __name__ == "__main__":
    out_dir = "frontend/samples"
    os.makedirs(out_dir, exist_ok=True)
    df = load_raw_dataframe()
    
    # Extract one sample per class
    for cls in CLASS_NAMES:
        cls_df = df[df["label"] == cls]
        if not cls_df.empty:
            sample = cls_df.iloc[0]["waferMap"]
            img = wafer_to_image(sample)
            # Resize it just a bit bigger for better display
            img = img.resize((128, 128), Image.NEAREST)
            safe_name = cls.lower().replace("-", "_") + ".png"
            img.save(os.path.join(out_dir, safe_name))
            print(f"Saved {safe_name}")
