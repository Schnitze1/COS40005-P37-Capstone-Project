module load apptainer

apptainer build containers/image.sif containers/image.def

# mujoco_py writes/compiles artifacts into mujoco_py/generated on first import.
# Use a writable tmpfs overlay; add --fakeroot so it has permission to write there.
apptainer exec --nv --writable-tmpfs --fakeroot \
  containers/image.sif \
  bash -lc '
    cd unlearning
    python mujoco_fully_training.py --dataset hopper-medium-expert-v0 --algo IQL --model ./params/iql_hopper_em_params.json
  '
