#! /usr/bin/bash
#SBATCH --job-name=Unet_array_fractography
#SBATCH --array=0-17%1  # This creates 18 tasks (0 to 17), run 3 at a time
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=3
#SBATCH --time 20:00:00
#SBATCH --gpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --output=%x/slurm_%j.out
#SBATCH --error=%x/slurm_%j.err
#SBATCH --account=rxf131
base_path="/home/aml334/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/fractography/Att_Unet_test"
nvidia-smi --query-gpu=gpu_name,driver_version,memory.total,memory.used --format=csv

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "epochs: $epochs" 
echo "accumulation steps: $accumulation_steps"
echo "encoder_pairs: $encoder_pairs"
echo "initial_features: $initial_features"
echo "input_channels: $input_channels"
echo "output_channels: $output_channels"
echo "final_activation: $final_activation"
echo "learning_rate: $learning_rate"
path="${base_path}/${i}"
echo "path: $path"
echo $SLURM_GPUS_ON_NODE
mkdir -p "$path"
/mnt/vstor/CSE_MSE_RXF131/sdle-ondemand/pioneer/config/run.sh --nv\
/home/rxf131/ondemand/share/build_link/apt_gpu-pt.sif\
python3\
-m\
torch.distributed.launch\
--nproc_per_node=2\
/home/aml334/CSE_MSE_RXF131/cradle-members/mds3/aml334/mds3-advman-2/topics/aml-fractography/fractography_scripts/fatigue_training.py\
300\
1\
3\
1\
1\
0.000005\
12\
$path > "${path}/output.txt" 2>&1

