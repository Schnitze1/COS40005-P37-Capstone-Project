import os
import torch

ckpt_path = "Weak_trained/kitchen-mixed-v0/CQL_20260817181810/model_250000.pt"

# 1. Basic file check
size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
print(f"File size: {size_mb:.2f} MB")

# 2. Try loading it
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Loaded OK")
    if isinstance(ckpt, dict):
        print("Keys:", list(ckpt.keys()))
except Exception as e:
    print(f"FAILED TO LOAD: {type(e).__name__}: {e}")
