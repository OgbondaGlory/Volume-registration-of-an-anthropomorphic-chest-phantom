#!/usr/bin/bash
#
# Project/Account (use your own)
#SBATCH -A scw2109
#SBATCH --job-name=segm     # Job name
#SBATCH --output segm-%j.out     # Job name
#SBATCH --error  segm-%j.err     # Job name
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
#SBATCH --time=1:00:00
#SBATCH --mem=5G

module load python/3.10.4 CUDA/11.7
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

    echo "Segmentation of $dataset"
    ./Deformable_Image_Registration.py $dataset Results/$dataset OPERATION segment
done
