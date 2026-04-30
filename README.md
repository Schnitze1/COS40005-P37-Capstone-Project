TrajDeleter: OzSTAR Execution Guide
This document provides the necessary commands and workflow to execute the Offline RL training and evaluation pipeline on the OzSTAR HPC cluster.

1. Environment Preparation
All dependencies are encapsulated in an Apptainer container to ensure consistency and bypass read-only filesystem restrictions.

Bash
# Navigate to the project directory
cd /fred/oz479/COS40005-P37-Capstone-Project/

# Load the required module
module load apptainer
2. Launching Training Jobs (Slurm)
To train the unlearning agent, submit the job script to the skylake-gpu partition. Ensure your batch script is configured for the GPU nodes.

Bash
# Submit the training job
sbatch scripts/train_unlearning.sh
To monitor job status:

``` Bash
# View active jobs for your user
squeue -u [your_username]

# View live logs (replace JOB_ID)
tail -f slurm-JOB_ID.out
3. Interactive Evaluation (MIA & Utility)
To run a specific model checkpoint (e.g., for verification or client demo), request an interactive GPU node first.
```
```
Bash
# Request 1 hour on a GPU node
salloc --partition=skylake-gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00
Once on the compute node, run the evaluation script using the Seeded Bind Mount to allow MuJoCo to compile its binaries:
```
```
Bash
# Run the performance test
apptainer exec --nv --bind ./mujoco_tmp_build:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env MUJOCO_GL=osmesa p37_env.sif \
  python3 TrajDeleter/unlearning/performance_test.py \
  --model [PATH_TO_JSON] \
  --model_params [PATH_TO_PT_FILE] \
  --task kitchen-mixed-v0
```
4. Key File Locations
Models/Logs: Mujoco_our_method_2000000_1.0/stage2/kitchen-mixed-v0-0/

Datasets: /fred/oz479/COS40005-P37-Capstone-Project/data/

Container: p37_env.sif
