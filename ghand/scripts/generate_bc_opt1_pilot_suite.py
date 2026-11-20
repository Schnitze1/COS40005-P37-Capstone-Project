import os
import json

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Single Consistent Base Run: BEAR_20260827180045 (Seed 42)
ckpt_dir = os.path.join(base_dir, "Fully_trained/kitchen-complete-v0/BEAR_20260827180045")
ckpt_m30k = os.path.join(ckpt_dir, "model_30000.pt")    # Eval Return: 2.40
ckpt_m40k = os.path.join(ckpt_dir, "model_40000.pt")    # Eval Return: 1.90
ckpt_m170k = os.path.join(ckpt_dir, "model_170000.pt")  # Eval Return: 1.50

perf_flags = "--enable_tf32 --batch_size 512"

pilot_experiments = [
    {
        "folder": "bc_pilot_lr1e4_s42",
        "job_name": "bc_p1_1e4",
        "desc": "PILOT-01: Standalone Pure BC Upper Bound Baseline (Actor LR 1e-4, Seed 42, 200k Steps)",
        "time": "02:00:00",
        "flag": f"--seed 42 --n_steps 200000 --n_steps_per_epoch 10000 --warmup_steps 200000 --actor_lr 1e-4 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps 200000 {perf_flags}"
    },
    {
        "folder": "opt1_pilot_m30k_lr5e5_s42",
        "job_name": "opt1_p2_5e5",
        "desc": "PILOT-02: Load Peak BC (model_30000.pt, Return 2.40), Moderate Fine-tune Actor LR 5e-5, Seed 42",
        "time": "02:00:00",
        "flag": f"--seed 42 --n_steps 200000 --n_steps_per_epoch 10000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 5e-5 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps 200000 --max_q_target 400.0 {perf_flags} --load_model {ckpt_m30k}"
    },
    {
        "folder": "opt1_pilot_m30k_lr1e4_s42",
        "job_name": "opt1_p3_1e4",
        "desc": "PILOT-03: Load Peak BC (model_30000.pt, Return 2.40), Fast Fine-tune Actor LR 1e-4, Seed 42",
        "time": "02:00:00",
        "flag": f"--seed 42 --n_steps 200000 --n_steps_per_epoch 10000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 1e-4 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps 200000 --max_q_target 400.0 {perf_flags} --load_model {ckpt_m30k}"
    },
    {
        "folder": "opt1_pilot_m40k_lr5e5_s42",
        "job_name": "opt1_p4_m40k",
        "desc": "PILOT-04: Load Early BC (model_40000.pt, Return 1.90), Fine-tune Actor LR 5e-5, Seed 42",
        "time": "02:00:00",
        "flag": f"--seed 42 --n_steps 200000 --n_steps_per_epoch 10000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 5e-5 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps 200000 --max_q_target 400.0 {perf_flags} --load_model {ckpt_m40k}"
    },
    {
        "folder": "opt1_pilot_m170k_lr5e5_s42",
        "job_name": "opt1_p5_m170k",
        "desc": "PILOT-05: Load Late BC (model_170000.pt, Return 1.50), Fine-tune Actor LR 5e-5, Seed 42",
        "time": "02:00:00",
        "flag": f"--seed 42 --n_steps 200000 --n_steps_per_epoch 10000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 5e-5 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps 200000 --max_q_target 400.0 {perf_flags} --load_model {ckpt_m170k}"
    }
]

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

echo "Running 5-Job Pilot Experiment: {desc}"
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

submit_all_content = "#!/bin/bash\n# Launch 5-Job Option 1 BEAR Pilot Suite\nset -e\n\n"

for exp in pilot_experiments:
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
        
    print(f"Configured pilot experiment folder: experiments/{exp['folder']}/")
    
    submit_all_content += f"echo 'Submitting {exp['job_name']}...'\n(cd experiments/{exp['folder']} && ./submit.sh)\nsleep 1\n\n"

submit_all_path = os.path.join(base_dir, "scripts", "submit_bc_opt1_pilot_suite.sh")
with open(submit_all_path, "w") as f:
    f.write(submit_all_content)
os.chmod(submit_all_path, 0o755)

print(f"\nGenerated master pilot submission script: scripts/submit_bc_opt1_pilot_suite.sh")
