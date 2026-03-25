"""
download_dataset.py
====================
Downloads and prepares the WM-811K dataset.

The WM-811K dataset can be downloaded from Kaggle.
This script helps you set up the Kaggle API and download it.

INSTRUCTIONS:
1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section, click "Create New Token"
3. A file called 'kaggle.json' will download
4. Run this script -- it will guide you through pasting the credentials

OR if you already have kaggle.json, drop it in:
    C:\\Users\\<YourName>\\.kaggle\\kaggle.json
then re-run this script.
"""

import os
import sys
import json
import shutil

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KAGGLE_DIR = os.path.join(os.path.expanduser("~"), ".kaggle")
KAGGLE_JSON = os.path.join(KAGGLE_DIR, "kaggle.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KAGGLE_DIR, exist_ok=True)

print("=" * 60)
print("  WaferMapAI - Phase 1: Dataset Download")
print("=" * 60)
print()

# ── Check if dataset already exists ──────────────────────────────────────────
pkl_path = os.path.join(DATA_DIR, "LSWMD.pkl")
if os.path.exists(pkl_path):
    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"[OK] Dataset already exists: {pkl_path} ({size_mb:.1f} MB)")
    print("     No download needed. You can run: python notebooks/explore_data.py")
    sys.exit(0)

# ── Check Kaggle JSON ─────────────────────────────────────────────────────────
if not os.path.exists(KAGGLE_JSON):
    print("[!!] Kaggle API token not found.")
    print()
    print("  How to get your Kaggle API token:")
    print("  1. Go to: https://www.kaggle.com/settings/account")
    print("  2. Scroll down to the 'API' section")
    print("  3. Click 'Create New API Token'")
    print("  4. A file 'kaggle.json' will download to your computer")
    print()
    print("  Paste the contents of kaggle.json below.")
    print("  It looks like: {\"username\":\"yourname\",\"key\":\"abc123...\"}")
    print()

    raw = input("  Paste kaggle.json contents here: ").strip()
    try:
        data = json.loads(raw)
        with open(KAGGLE_JSON, "w") as f:
            json.dump(data, f)
        # restrict permissions (Kaggle requires this on Linux/Mac, optional on Windows)
        print(f"[OK] Saved Kaggle credentials to {KAGGLE_JSON}")
    except json.JSONDecodeError:
        print("[ERROR] That doesn't look like valid JSON. Please try again.")
        sys.exit(1)

# ── Download via Kaggle API ───────────────────────────────────────────────────
print()
print("[Download] Downloading WM-811K dataset from Kaggle...")
print("           This is ~900MB and may take a few minutes.")
print()

import subprocess
result = subprocess.run(
    [sys.executable, "-m", "kaggle", "datasets", "download",
     "-d", "qingyi/wm811k-wafer-map",
     "-p", DATA_DIR, "--unzip"],
    capture_output=False
)

if result.returncode != 0:
    print()
    print("[ERROR] Download failed. Possible reasons:")
    print("  1. Invalid Kaggle credentials")
    print("  2. You have not accepted the dataset's terms on Kaggle")
    print("     -> Visit https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
    print("        and click 'Download' once manually to accept terms.")
    sys.exit(1)

# ── Verify ────────────────────────────────────────────────────────────────────
if os.path.exists(pkl_path):
    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print()
    print(f"[OK] Dataset ready: {pkl_path} ({size_mb:.1f} MB)")
    print()
    print("  You can now run: python notebooks/explore_data.py")
else:
    # Look for any pkl file in data/
    pks = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
    if pks:
        print(f"[OK] Found PKL files in data/: {pks}")
        print("     Please rename to LSWMD.pkl if different, then run explore_data.py")
    else:
        print("[!!] No .pkl file found in data/. Check any .zip that downloaded.")
