To run the tests, I've been using the commands

~~~
module load apptainer

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS2_ASSET_DIR=/fred/oz479/COS40005-P37-Capstone-Project/maniskill_assets

apptainer exec --nv \
    -B /fred/oz479 \
    --env VK_ICD_FILENAMES=$VK_ICD_FILENAMES \
    --env MS2_ASSET_DIR=$MS2_ASSET_DIR \
    --env DISPLAY='' \
    /fred/oz479/COS40005-P37-Capstone-Project/p37_env_with_maniskill.sif \
    python3 /fred/oz479/jburns/maniskill2_work/test_maniskill.py
~~~

For the training script, it takes a number of params

--task
--dataset_path (This is present as I wasn't sure if we had a Maniskill2 dataset yet or not)
--algo 
--gpu

for now, I think the run command at the bottom of the slurm file should work (Changed around for your local filesystem)