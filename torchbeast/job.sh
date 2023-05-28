#!/bin/bash

# Set the number of tasks and CPUs per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Set the maximum time and memory allowed
#SBATCH --time=12:00:00
#SBATCH --mem=4G

# Activate the Python environment, if necessary
source activate torchbeast

# Parse any additional flags as arguments to the Python process
# shift
flags=""
while [ $# -gt 0 ]; do
    flags="$flags $1"
    shift
done

# Run the Python process with the specified flags
python monobeast.py $flags
