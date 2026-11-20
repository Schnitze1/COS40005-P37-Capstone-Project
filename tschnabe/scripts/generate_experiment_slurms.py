import os

base_dir = os.path.dirname(os.path.abspath(__file__))

experiments = [
    {
        "folder": "exp0a_bear_baseline_seed42",
        "job_name": "bear_exp0a_s42",
        "desc": "Exp 0a: Baseline BEAR (Seed 42)",
        "time": "06:00:00",
        "flag": "--seed 42"
    },
    {
        "folder": "exp0b_bear_baseline_seed43",
        "job_name": "bear_exp0b_s43",
        "desc": "Exp 0b: Baseline BEAR Verification (Seed 43)",
        "time": "06:00:00",
        "flag": "--seed 43"
    },
    {
        "folder": "exp5a_bear_qclip400_seed42",
        "job_name": "bear_exp5a_s42",
        "desc": "Exp 5a: Soft Target Q-Bounding Q_max=400 (Seed 42)",
        "time": "06:00:00",
        "flag": "--seed 42 --max_q_target 400.0"
    },
    {
        "folder": "exp5b_bear_qclip400_seed43",
        "job_name": "bear_exp5b_s43",
        "desc": "Exp 5b: Soft Target Q-Bounding Q_max=400 Verification (Seed 43)",
        "time": "06:00:00",
        "flag": "--seed 43 --max_q_target 400.0"
    },
    {
        "folder": "exp6a_bear_warmup100k_seed42",
        "job_name": "bear_exp6a_s42",
        "desc": "Exp 6a: Extended Warmup 100k (Seed 42)",
        "time": "06:00:00",
        "flag": "--seed 42 --warmup_steps 100000"
    },
    {
        "folder": "exp6b_bear_warmup100k_seed43",
        "job_name": "bear_exp6b_s43",
        "desc": "Exp 6b: Extended Warmup 100k Verification (Seed 43)",
        "time": "06:00:00",
        "flag": "--seed 43 --warmup_steps 100000"
    },
    {
        "folder": "exp9a_bear_800k_hybrid_seed42",
        "job_name": "bear_exp9a_s42",
        "desc": "Exp 9a: 800k Step Hybrid BEAR with Actor BC Warmup (Seed 42)",
        "time": "12:00:00",
        "flag": "--seed 42 --n_steps 800000 --n_steps_per_epoch 10000 --warmup_steps 400000 --q_ramp_steps 50000 --actor_lr_post_warmup 2e-5 --alpha_min 0.1 --max_q_target 400.0 --ramp_type linear --bc_warmup_weight 1.0"
    },
    {
        "folder": "exp9b_bear_800k_hybrid_seed43",
        "job_name": "bear_exp9b_s43",
        "desc": "Exp 9b: 800k Step Hybrid BEAR with Actor BC Warmup Verification (Seed 43)",
        "time": "12:00:00",
        "flag": "--seed 43 --n_steps 800000 --n_steps_per_epoch 10000 --warmup_steps 400000 --q_ramp_steps 50000 --actor_lr_post_warmup 2e-5 --alpha_min 0.1 --max_q_target 400.0 --ramp_type linear --bc_warmup_weight 1.0"
    }
]

slurm_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --partition=skylake-gpu,milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time={time_limit}

module load apptainer
export CUDA_VISIBLE_DEVICES=0

if [ -n "${{SLURM_SUBMIT_DIR}}" ]; then
  if [ -d "${{SLURM_SUBMIT_DIR}}/TrajDeleter" ]; then
    WS_DIR="${{SLURM_SUBMIT_DIR}}"
  elif [ -d "${{SLURM_SUBMIT_DIR}}/../TrajDeleter" ]; then
    WS_DIR="$(cd "${{SLURM_SUBMIT_DIR}}/.." && pwd)"
  elif [ -d "${{SLURM_SUBMIT_DIR}}/../../TrajDeleter" ]; then
    WS_DIR="$(cd "${{SLURM_SUBMIT_DIR}}/../.." && pwd)"
  else
    WS_DIR="/fred/oz479/COS40005-P37-Capstone-Project/${{USER}}"
  fi
else
  WS_DIR="/fred/oz479/COS40005-P37-Capstone-Project/${{USER}}"
fi

GEN_DIR="/tmp/${{USER}}_mujoco_gen_${{SLURM_JOB_ID}}"
PATCH_DIR="/tmp/${{USER}}_patch_${{SLURM_JOB_ID}}"

mkdir -p $GEN_DIR
apptainer exec $WS_DIR/p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

export D4RL_DATASET_DIR="$WS_DIR/data"

mkdir -p $PATCH_DIR
cp $WS_DIR/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

mkdir -p $WS_DIR/logs

echo "Running Experiment: {desc}"

apptainer exec --nv --writable-tmpfs --ipc \\
  -B /fred/oz479 \\
  -B ${{GEN_DIR}}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \\
  -B ${{PATCH_DIR}}/torch_utility.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/torch_utility.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/algos/bear.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/bear.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/algos/torch/bear_impl.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/torch/bear_impl.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/algos/torch/ddpg_impl.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/torch/ddpg_impl.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/models/torch/q_functions/mean_q_function.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/models/torch/q_functions/mean_q_function.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/models/torch/q_functions/ensemble_q_function.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/models/torch/q_functions/ensemble_q_function.py \\
  -B ${{WS_DIR}}/TrajDeleter/packages/d3rlpy/metrics/scorer.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/metrics/scorer.py \\
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \\
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \\
  --env MUJOCO_GL=osmesa \\
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \\
  $WS_DIR/p37_env.sif python3 $WS_DIR/TrajDeleter/unlearning/mujoco_fully_training.py \\
  --dataset kitchen-complete-v0 \\
  --algo BEAR \\
  --n_steps 500000 \\
  --n_steps_per_epoch 50000 \\
  --n_critics 4 \\
  --lam 0.50 \\
  --n_action_samples 10 \\
  --n_mmd_action_samples 10 \\
  --alpha_threshold 0.15 \\
  --alpha_lr 1e-4 \\
  --initial_alpha 1.0 \\
  --mmd_kernel kl \\
  --mmd_sigma 20.0 \\
  --scaler standard \\
  --actor_lr 1e-4 \\
  --critic_lr 5e-5 \\
  --reward_scale 0.25 \\
  --use_huber_loss \\
  --huber_delta 1.0 \\
  --warmup_steps 40000 \\
  --gpu 0 \\
  {flag}

rm -rf $PATCH_DIR $GEN_DIR
"""

for exp in experiments:
    folder_path = os.path.join(base_dir, "experiments", exp["folder"])
    os.makedirs(folder_path, exist_ok=True)
    slurm_path = os.path.join(folder_path, f"run_{exp['folder']}.slurm")
    
    content = slurm_template.format(
        job_name=exp["job_name"],
        time_limit=exp["time"],
        desc=exp["desc"],
        flag=exp["flag"]
    )
    
    with open(slurm_path, "w") as f:
        f.write(content)
    print(f"Generated {slurm_path}")

print(f"All {len(experiments)} experiment Slurm scripts generated successfully!")
