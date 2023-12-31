#!/usr/bin/bash
#
#SBATCH --mail-user=glory.ogbonda@bangor.ac.uk
# Project/Account (use your own)
#SBATCH -A scw2109
#SBATCH --job-name=apply_transforms     # Job name
#SBATCH --output apply_transforms-%j.out # Standard output
#SBATCH --error  apply_transforms-%j.err # Standard error
#
# Number of tasks per node
#SBATCH --ntasks-per-node=1
#
# Number of cores per task
#SBATCH --cpus-per-task=20
#
# Use one node
#SBATCH --nodes=1
#
# Runtime of this jobs is less than 6 hours.
#SBATCH --time=00:60:00
#SBATCH --mem=65G

# Load required modules
module load python/3.10.4 CUDA/11.7
# Assuming required Python packages are already installed

# Set matplotlib backend if required
export MPLBACKEND=pdf


# Define the path to the labels directory and .dat file

LABELS_DIRECTORY="Phantom_CT_Scan_Segmentation/"
DAT_FILE_PATH="Phantom_CT_Scan_Segmentation/map.dat"

for dataset in "Patient_CT_Scan_1" "Patient_CT_Scan_2" "Patient_CT_Scan_3" "Patient_CT_Scan_4"
do
    mkdir -p Results/$dataset
    echo "Applying transformations for $dataset"
    python3 ./apply_transformations.py $dataset $LABELS_DIRECTORY $DAT_FILE_PATH
done