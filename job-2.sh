#!/usr/bin/bash
#
# Project/Account (use your own)
#SBATCH -A scw2109
#SBATCH --job-name=recons     # Job name
#SBATCH --output recons-%j.out     # Job name
#SBATCH --error  recons-%j.err     # Job name
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
# Runtime of this jobs is less than 5 hours.
#SBATCH --time=72:00:00
#SBATCH --mem=10G

#source ./modules.sh
#module purge > /dev/null 2>&1
#module load cmake compiler/gnu/8/1.0 CUDA python/3.7.0


module load python/3.10.4 CUDA/11.7
#python3 -m pip install --user matplotlib SimpleITK
#python3 -m pip install --user voxelmorph



export MPLBACKEND=pdf

if [ ! -d Results ]
then
   mkdir Results
fi

for dataset in "Patient_CT_Scan_2" # "Patient_CT_Scan_2" "Patient_CT_Scan_3" "Patient_CT_Scan_4"
do

   if [ ! -d Results/$dataset ]
   then
      mkdir Results/$dataset
   fi

    echo "Registration of $dataset"
    ./Deformable_Image_Registration.py $dataset Results/$dataset 
done

