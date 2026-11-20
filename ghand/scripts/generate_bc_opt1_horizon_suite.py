import os
import json

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Base Run: BEAR_20260827180045 (Seed 42)
ckpt_dir = os.path.join(base_dir, "Fully_trained/kitchen-complete-v0/BEAR_20260827180045")
ckpt_m30k = os.path.join(ckpt_dir, "model_30000.pt")    # Return: 2.40
ckpt_m40k = os.path.join(ckpt_dir, "model_40000.pt")    # Return: 1.90
ckpt_m170k = os.path.join(ckpt_dir, "model_170000.pt")  # Return: 1.50

perf_flags = "--enable_tf32 --batch_size 512"

# Define 5 core setups
configs = [
    {
        "id": "bc_scratch",
        "name_prefix": "bc",
        "name_suffix": "lr1e4_s42",
        "job_code": "bc_1e4",
        "desc_base": "Pure BC Baseline (Actor LR 1e-4, Seed 42)",
        "flag_func": lambda steps, wall_time: f"--seed 42 --n_steps {steps} --n_steps_per_epoch 20000 --warmup_steps {steps} --actor_lr 1e-4 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps {steps} {perf_flags}"
    },
    {
        "id": "opt1_m30k_5e5",
        "name_prefix": "opt1",
        "name_suffix": "m30k_lr5e5_s42",
        "job_code": "o1_30k_5e5",
        "desc_base": "Option 1 (model_30000.pt Return 2.40, Actor LR 5e-5, Seed 42)",
        "flag_func": lambda steps, wall_time: f"--seed 42 --n_steps {steps} --n_steps_per_epoch 20000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 5e-5 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps {steps} --max_q_target 400.0 {perf_flags} --load_model {ckpt_m30k}"
    },
    {
        "id": "opt1_m30k_1e4",
        "name_prefix": "opt1",
        "name_suffix": "m30k_lr1e4_s42",
        "job_code": "o1_30k_1e4",
        "desc_base": "Option 1 (model_30000.pt Return 2.40, Actor LR 1e-4, Seed 42)",
        "flag_func": lambda steps, wall_time: f"--seed 42 --n_steps {steps} --n_steps_per_epoch 20000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 1e-4 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps {steps} --max_q_target 400.0 {perf_flags} --load_model {ckpt_m30k}"
    },
    {
        "id": "opt1_m40k_5e5",
        "name_prefix": "opt1",
        "name_suffix": "m40k_lr5e5_s42",
        "job_code": "o1_40k_5e5",
        "desc_base": "Option 1 (model_40000.pt Return 1.90, Actor LR 5e-5, Seed 42)",
        "flag_func": lambda steps, wall_time: f"--seed 42 --n_steps {steps} --n_steps_per_epoch 20000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 5e-5 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps {steps} --max_q_target 400.0 {perf_flags} --load_model {ckpt_m40k}"
    },
    {
        "id": "opt1_m170k_5e5",
        "name_prefix": "opt1",
        "name_suffix": "m170k_lr5e5_s42",
        "job_code": "o1_170k_5e5",
        "desc_base": "Option 1 (model_170000.pt Return 1.50, Actor LR 5e-5, Seed 42)",
        "flag_func": lambda steps, wall_time: f"--seed 42 --n_steps {steps} --n_steps_per_epoch 20000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 5e-5 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps {steps} --max_q_target 400.0 {perf_flags} --load_model {ckpt_m170k}"
    }
]

# Empirical timing: 10,000 steps = 4 minutes.
# 500k steps = 3.33 hours -> Allocated 04:00:00 (20% safety margin)
# 800k steps = 5.33 hours -> Allocated 06:00:00 (15% safety margin)
# 1M steps   = 6.67 hours -> Allocated 07:30:00 (15% safety margin)
horizons = [
    {"tag": "500k", "steps": 500000, "time": "04:00:00"},
    {"tag": "800k", "steps": 800000, "time": "06:00:00"},
    {"tag": "1m", "steps": 1000000, "time": "07:30:00"}
]

horizon_experiments = []

for h in horizons:
    for cfg in configs:
        folder_name = f"{cfg['name_prefix']}_{h['tag']}_{cfg['name_suffix']}"
        job_name = f"{cfg['job_code']}_{h['tag']}"
        desc = f"{cfg['desc_base']} [{h['tag'].upper()} Steps]"
        flags = cfg["flag_func"](h["steps"], h["time"])
        
        horizon_experiments.append({
            "folder": folder_name,
            "job_name": job_name,
            "desc": desc,
            "time": h["time"],
            "flag": flags,
            "horizon": h["tag"]
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

echo "Running Step Horizon Variation Experiment: {desc}"
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

submit_all_content = "#!/bin/bash\n# Submit 15-Job Step Horizon Variation Suite (500k, 800k, 1M)\nset -e\n\n"

for exp in horizon_experiments:
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
        
    print(f"Configured horizon experiment folder: experiments/{exp['folder']}/")
    
    submit_all_content += f"echo 'Submitting {exp['job_name']}...'\n(cd experiments/{exp['folder']} && ./submit.sh)\nsleep 1\n\n"

submit_all_path = os.path.join(base_dir, "scripts", "submit_bc_opt1_horizon_suite.sh")
with open(submit_all_path, "w") as f:
    f.write(submit_all_content)
os.chmod(submit_all_path, 0o755)

print(f"\nGenerated master horizon submission script: scripts/submit_bc_opt1_horizon_suite.sh")
