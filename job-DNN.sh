#!/usr/bin/bash
#
# Project/Account (use your own)
#SBATCH -A scw2109
#SBATCH --job-name=DNN     # Job name
#SBATCH --output DNN-%j.out     # Job name
#SBATCH --error  DNN-%j.err     # Job name
#
# Number of tasks per node
#SBATCH --ntasks-per-node=1
#
# Number of cores per task
#SBATCH --cpus-per-task=40
#
# Use one node
#SBATCH --nodes=1
#
# We ask for 1 tasks with 1 core only.
# We ask for a GPU
#SBATCH -p gpu_v100
#SBATCH --gres=gpu:2
#
# Runtime of this jobs is less than 5 hours.
#SBATCH --time=48:00:00
#SBATCH --mem=80G

#source ./modules.sh
#module purge > /dev/null 2>&1
#module load cmake compiler/gnu/8/1.0 CUDA python/3.7.0


module load python/3.10.4
module load CUDA/11.7

python3 -m pip install --user matplotlib SimpleITK
python3 -m pip install --user torch scipy voxelmorph



export MPLBACKEND=pdf

INPUT_DIR="4.000000-Lung 1.0-09229"
OUTPUT_DIR="Result-4.000000-Lung 1.0-09229"

date > $OUTPUT_DIR/recons_runtime
./Deformable_Image_Registration.py  $INPUT_DIR $OUTPUT_DIR dnn
date >> $OUTPUT_DIR/recons_runtime
