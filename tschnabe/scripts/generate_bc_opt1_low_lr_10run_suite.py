import os
import json

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Peak BC Checkpoint (Eval Return: 2.40)
ckpt_m30k = os.path.join(base_dir, "Fully_trained/kitchen-complete-v0/BEAR_20260827180045/model_30000.pt")

perf_flags = "--enable_tf32 --batch_size 512 --use_huber_loss"

seeds = [42, 43, 44, 45, 46]
lrs = [
    {"lr_str": "5e-6", "val": "5e-6", "code": "5e6"},
    {"lr_str": "1e-6", "val": "1e-6", "code": "1e6"}
]

experiments = []

for lr in lrs:
    for s in seeds:
        folder_name = f"opt1_1m_m30k_lr{lr['code']}_s{s}"
        job_name = f"o1_30k_{lr['code']}_s{s}"
        desc = f"1M Option 1 fine-tuning: Load model_30000.pt (Return 2.40), Actor LR {lr['lr_str']}, Seed {s}, Eval per 10k steps"
        flag = f"--seed {s} --n_steps 1000000 --n_steps_per_epoch 10000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr {lr['val']} --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps 1000000 --max_q_target 400.0 {perf_flags} --load_model {ckpt_m30k}"
        
        experiments.append({
            "folder": folder_name,
            "job_name": job_name,
            "desc": desc,
            "time": "07:30:00",
            "flag": flag,
            "seed": s,
            "actor_lr": lr["val"]
        })

slurm_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={ws_dir}/experiments/{folder}/logs/{job_name}_%j.out
#SBATCH --error={ws_dir}/experiments/{folder}/logs/{job_name}_%j.err
#SBATCH --time={time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1

module load apptainer
export CUDA_VISIBLE_DEVICES=0

WS_DIR="/fred/oz479/COS40005-P37-Capstone-Project/tschnabe"
GEN_DIR="/tmp/${{USER}}_mujoco_gen_${{SLURM_JOB_ID}}"
DATA_TMPDIR="/tmp/${{USER}}_d4rl_data_${{SLURM_JOB_ID}}"

mkdir -p $GEN_DIR
apptainer exec $WS_DIR/p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

# Copy D4RL dataset to local scratch for faster I/O
mkdir -p $DATA_TMPDIR
if [ -f "$WS_DIR/data/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" ]; then
    cp "$WS_DIR/data/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" "$DATA_TMPDIR/" 2>/dev/null || true
elif [ -f "/home/tschnabe/.d4rl/datasets/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" ]; then
    cp "/home/tschnabe/.d4rl/datasets/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" "$DATA_TMPDIR/" 2>/dev/null || true
fi
export D4RL_DATASET_DIR="$DATA_TMPDIR"

mkdir -p {ws_dir}/experiments/{folder}/logs

echo "Running 1M Low-LR Option 1 Experiment: {desc}"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start Time: $(date)"

apptainer exec --nv --writable-tmpfs --ipc \\
  -B /fred/oz479 \\
  -B /home/tschnabe \\
  -B /tmp \\
  -B ${{GEN_DIR}}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/algos/bear.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/bear.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/algos/torch/bear_impl.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/torch/bear_impl.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/algos/torch/sac_impl.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/torch/sac_impl.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/algos/torch/ddpg_impl.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/torch/ddpg_impl.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/models/torch/q_functions/mean_q_function.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/models/torch/q_functions/mean_q_function.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/models/torch/q_functions/ensemble_q_function.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/models/torch/q_functions/ensemble_q_function.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/metrics/scorer.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/metrics/scorer.py \\
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \\
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \\
  --env MUJOCO_GL=osmesa \\
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \\
  --env PYTHONUNBUFFERED=1 \\
  $WS_DIR/p37_env.sif python3 -u $WS_DIR/TrajDeleter/unlearning/mujoco_fully_training.py \\
  --dataset kitchen-complete-v0 \\
  --algo BEAR \\
  --n_critics 4 \\
  --lam 0.50 \\
  --n_action_samples 10 \\
  --n_mmd_action_samples 10 \\
  --alpha_threshold 0.15 \\
  {flag}

# Cleanup local scratch
rm -rf $DATA_TMPDIR $GEN_DIR

echo "End Time: $(date)"
"""

experiments_dir = os.path.join(base_dir, "experiments")
os.makedirs(experiments_dir, exist_ok=True)

submit_all_content = "#!/bin/bash\n# Launch 10-Job Low-LR 1M Option 1 Benchmark Suite\nset -e\n\n"

for exp in experiments:
    exp_folder_dir = os.path.join(experiments_dir, exp['folder'])
    logs_dir = os.path.join(exp_folder_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    script_name = f"{exp['folder']}.slurm"
    script_path = os.path.join(exp_folder_dir, script_name)
    
    formatted_slurm = slurm_template.format(
        job_name=exp["job_name"],
        time=exp["time"],
        desc=exp["desc"],
        flag=exp["flag"],
        ws_dir=base_dir,
        folder=exp["folder"]
    )
    
    with open(script_path, "w") as f:
        f.write(formatted_slurm)
        
    local_submit_path = os.path.join(exp_folder_dir, "submit.sh")
    with open(local_submit_path, "w") as f:
        f.write(f"#!/bin/bash\nsbatch {script_name}\n")
    os.chmod(local_submit_path, 0o755)
    
    info_path = os.path.join(exp_folder_dir, "experiment_info.json")
    with open(info_path, "w") as f:
        json.dump(exp, f, indent=2)
        
    print(f"Configured low-LR 1M experiment folder: experiments/{exp['folder']}/")
    
    submit_all_content += f"echo 'Submitting {exp['job_name']}...'\n(cd experiments/{exp['folder']} && ./submit.sh)\nsleep 1\n\n"

submit_all_path = os.path.join(base_dir, "scripts", "submit_bc_opt1_low_lr_10run_suite.sh")
with open(submit_all_path, "w") as f:
    f.write(submit_all_content)
os.chmod(submit_all_path, 0o755)

print(f"\nGenerated master low-LR 1M submission script: scripts/submit_bc_opt1_low_lr_10run_suite.sh")
