#!/bin/bash -l
#SBATCH -A cfn307359
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=long
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --job-name=XAS

conda activate py3.9
python3 run.py
