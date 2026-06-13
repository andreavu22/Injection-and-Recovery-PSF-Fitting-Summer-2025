#!/bin/bash
# script by Matthew
# Loic added ability to have an input filename when calling it


#SBATCH -J andrea_doublepsf_test_development # Job name
#SBATCH -o /work/10875/andreavu/ls6/injection_and_recovery_F1000W/job_running_output/o%j # Stdout output file
#SBATCH -e /work/10875/andreavu/ls6/injection_and_recovery_F1000W/job_running_output/e%j # Stdout output file/e%j # Stderr error file
#SBATCH -p development # Queue (partition) name
#SBATCH -N 1 # Total # of nodes
#SBATCH -n 1 # Total # of MPI tasks
#SBATCH -t 00:30:00 # Run time (hh:mm:ss)
#SBATCH --mail-type=all # Send email at begin and end of job
#SBATCH --mail-user=a24vu@uwaterloo.ca
#SBATCH -A AST24030


echo "uncomment these when running on TACC"
echo "module load python3/3.9.7"
module load python3/3.9.7

ibrun python rearrange_output_files.py

