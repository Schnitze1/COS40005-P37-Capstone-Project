import torch
import sys

path = "Fully_trained/kitchen-mixed-v0/CQL_20260402234703/model_1000.pt"
print(f"Patching {path}...")
# Load to CPU to strip all CUDA string metadata
data = torch.load(path, map_location='cpu')
# Force save it without specific device info
torch.save(data, "model_1000_cleaned.pt")
print("Cleaned weights saved to model_1000_cleaned.pt")
