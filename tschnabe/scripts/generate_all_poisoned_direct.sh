#!/bin/bash
set -e

echo "Regenerating all 5 poisoned datasets directly from the 3680-row HDF5 file..."

SOURCE="data/mini_kitchen_microwave_kettle_light_slider-v0.hdf5"
WEAK_JSON="Weak_trained/kitchen-complete-v0/CQL_20260821222942/params.json"
WEAK_PT="Weak_trained/kitchen-complete-v0/CQL_20260821222942/model_500000.pt"

mkdir -p /tmp/${USER}_d3rlpy_patch
cp $PWD/TrajDeleter/packages/d3rlpy/torch_utility.py /tmp/${USER}_d3rlpy_patch/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' /tmp/${USER}_d3rlpy_patch/torch_utility.py

CMD="apptainer exec --nv --writable-tmpfs --ipc -B /fred/oz479 -B /tmp -B /tmp/${USER}_d3rlpy_patch/torch_utility.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/torch_utility.py ./p37_env.sif python3 scripts/poison_kitchen_baffle_direct.py"

for rate_str in "01" "05" "10" "15" "20"; do
    rate_float=$(echo "scale=2; $rate_str / 100" | bc)
    out_file="data/kitchen_complete_baffle_${rate_str}_v2.hdf5"
    
    echo ""
    echo "========================================================="
    echo "Generating $rate_str% poisoned dataset -> $out_file"
    echo "========================================================="
    $CMD \
        --source_hdf5 $SOURCE \
        --weak_model_json $WEAK_JSON \
        --weak_model_pt $WEAK_PT \
        --poison_rate $rate_float \
        --output $out_file \
        --seed 42
done

echo ""
echo "Done regenerating all 5 poisoned datasets."
