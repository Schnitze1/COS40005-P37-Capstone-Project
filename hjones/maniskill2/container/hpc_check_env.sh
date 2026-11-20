#!/bin/bash
# HPC Environment Check Script
# Run this on OzSTAR to gather information before deployment

echo "===== HPC ENVIRONMENT CHECK ====="
echo ""

echo "1. Current Location:"
pwd
echo ""

echo "2. Home Directory:"
echo $HOME
echo ""

echo "3. Available Modules:"
module avail 2>&1 | grep -i apptainer
echo ""

echo "4. GPU Information (if on GPU node):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Not on GPU node or nvidia-smi not available"
echo ""

echo "5. GLIBC Version:"
ldd --version | head -n1
echo ""

echo "6. Python3 Available:"
which python3
python3 --version
echo ""

echo "7. Disk Usage in /fred/oz479:"
df -h /fred/oz479 2>/dev/null || echo "/fred/oz479 not accessible"
echo ""

echo "8. Check Project Directories:"
PROJECT=/fred/oz479/COS40005-P37-Capstone-Project
if [ -d "$PROJECT" ]; then
    echo "✓ Project directory exists: $PROJECT"
    ls -lh $PROJECT/
else
    echo "✗ Project directory NOT FOUND: $PROJECT"
fi
echo ""

echo "9. Check ManiSkill Assets:"
ASSETS=/fred/oz479/COS40005-P37-Capstone-Project/MANISKILL2/assets
if [ -d "$ASSETS" ]; then
    echo "✓ ManiSkill assets exists: $ASSETS"
    du -sh $ASSETS
else
    echo "✗ ManiSkill assets NOT FOUND: $ASSETS"
fi
echo ""

echo "10. Vulkan ICD (for GPU rendering):"
if [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    echo "✓ NVIDIA Vulkan ICD found"
    cat /etc/vulkan/icd.d/nvidia_icd.json
else
    echo "⚠ NVIDIA Vulkan ICD not found (expected on compute nodes with --nv)"
fi
echo ""

echo "===== ENVIRONMENT CHECK COMPLETE ====="
