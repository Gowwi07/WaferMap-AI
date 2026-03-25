"""
setup_check.py - Phase 0 verification script.
Run this to confirm every dependency is installed correctly.
"""
import sys
import importlib

print("=" * 60)
print(" WaferMapAI - Phase 0 Setup Verification")
print("=" * 60)
print(f"\n Python Version: {sys.version}\n")

checks = [
    ("torch",        "PyTorch         - Deep learning framework"),
    ("torchvision",  "TorchVision     - Image datasets & transforms"),
    ("numpy",        "NumPy           - Numerical computation"),
    ("pandas",       "Pandas          - Data analysis"),
    ("sklearn",      "Scikit-learn    - ML metrics & utilities"),
    ("matplotlib",   "Matplotlib      - Plotting"),
    ("seaborn",      "Seaborn         - Statistical plots"),
    ("cv2",          "OpenCV          - Image processing"),
    ("PIL",          "Pillow          - Image I/O"),
    ("streamlit",    "Streamlit       - Web dashboard"),
    ("tqdm",         "TQDM            - Progress bars"),
    ("scipy",        "SciPy           - Scientific computation"),
]

passed = 0
failed = []

for module, description in checks:
    try:
        importlib.import_module(module)
        print(f"  [OK]  {description}")
        passed += 1
    except ImportError:
        print(f"  [!!]  {description}  --> NOT INSTALLED")
        failed.append(module)

print()
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"  [GPU] GPU Detected: {gpu}")
        print(f"  [GPU] CUDA Version: {torch.version.cuda}")
    else:
        print("  [CPU] No GPU - will run on CPU (slower but fine for learning)")
    print(f"  [>>]  PyTorch Version: {torch.__version__}")
except Exception:
    pass

print()
print("=" * 60)
if not failed:
    print(f"  ALL {passed}/{passed} CHECKS PASSED -- Ready for Phase 1!")
else:
    print(f"  {len(failed)} package(s) missing: {', '.join(failed)}")
    print(f"  Run: pip install {' '.join(failed)}")
print("=" * 60)
