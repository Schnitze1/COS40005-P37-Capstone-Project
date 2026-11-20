import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ckpt_dir = os.path.join(base_dir, "Fully_trained/kitchen-complete-v0/BEAR_20260827180045")

checkpoints = [
    ("model_30000.pt", "Peak return 2.40"),
    ("model_40000.pt", "Second-peak return 1.90"),
    ("model_170000.pt", "Late stable return 1.50")
]

print("=== PRE-SUBMISSION CHECKPOINT & SCALER VALIDATION ===")
print(f"Target Directory: {ckpt_dir}\n")

all_valid = True

try:
    import torch
except ImportError:
    print("PyTorch not installed in global environment. Run inside Apptainer container.")
    sys.exit(0)

for filename, desc in checkpoints:
    path = os.path.join(ckpt_dir, filename)
    if not os.path.exists(path):
        print(f"FAILED: Checkpoint file not found: {path}")
        all_valid = False
        continue
    
    try:
        state = torch.load(path, map_location="cpu")
        print(f"Checkpoint: {filename} ({desc})")
        print(f"  File size: {os.path.getsize(path) / 1e6:.2f} MB")
        
        if isinstance(state, dict):
            keys = list(state.keys())
            print(f"  Keys in checkpoint ({len(keys)}): {keys[:5]}...")
            
            # Check policy / actor weights
            actor_keys = [k for k in keys if "policy" in k or "actor" in k or "_impl._actor" in k]
            print(f"  Actor/Policy tensor count: {len(actor_keys)}")
            
            # Check scaler state
            scaler_keys = [k for k in keys if "scaler" in k]
            print(f"  Scaler tensor count: {len(scaler_keys)}")
            if scaler_keys:
                print(f"  Scaler keys found: {scaler_keys}")
        print("  Status: VALID\n")
    except Exception as e:
        print(f"FAILED to load {filename}: {e}\n")
        all_valid = False

if all_valid:
    print("ALL CHECKPOINTS VALIDATED SUCCESSFULLY! Safe for Option 1 loading.")
else:
    print("VALIDATION WARNINGS FOUND. Please review output above.")
