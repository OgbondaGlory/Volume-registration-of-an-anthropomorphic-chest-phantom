#!/bin/bash
#
# Project/Account (use your own)
#SBATCH -A scw2109
#SBATCH --job-name=bspline     # Job name
#SBATCH --output bspline-%j.out     # Job name
#SBATCH --error  bspline-%j.err     # Job name
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
#SBATCH -p highmem
#
# Runtime of this jobs is less than 5 hours.
#SBATCH --time=28:00:00
#SBATCH --mem=30G

#source ./modules.sh
#module purge > /dev/null 2>&1
#module load cmake compiler/gnu/8/1.0 CUDA python/3.7.0


module load python/3.10.4 CUDA/11.7

#python3 -m pip install --user matplotlib SimpleITK
#python3 -m pip install --user voxelmorph
#python3 -m pip install tensorflow
# Download the TensorRT package from NVIDIA's website
wget https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2/local_repo/nv-tensorrt-repo-ubuntu1804-cuda11.0-trt7.2.1.6-ga-20201016_1-1_amd64.deb

# Install the downloaded package
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.0-trt7.2.1.6-ga-20201016_1-1_amd64.deb

# Add the necessary keys
sudo apt-key add /var/nv-tensorrt-repo-cuda11.0-trt7.2.1.6-ga-20201016/7fa2af80.pub

# Update the APT package repository cache
sudo apt-get update

# Install TensorRT
sudo apt-get install tensorrt

export MPLBACKEND=pdf

if [ ! -d Results ]
then
   mkdir Results
fi

for dataset in "Patient_CT_Scan_1" "Patient_CT_Scan_2" "Patient_CT_Scan_3" "Patient_CT_Scan_4"
do
   if [ ! -d Results/$dataset ]
   then
      mkdir Results/$dataset
   fi

    echo "B-spline Registration of $dataset"
    ./Deformable_Image_Registration.py $dataset Results/$dataset bspline
done
