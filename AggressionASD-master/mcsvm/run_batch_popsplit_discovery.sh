#!/bin/bash

#SBATCH --job-name=popsplits                # sets the job name
#SBATCH --cpus-per-task=1                         # sets 1 core for each task
#SBATCH --mem=10Gb                               # reserves 100 GB memory
#SBATCH --partition=ioannidis                  # requests that the job is executed in partition my partition
#SBATCH --time=100:00:00                            # reserves machines/cores for 4 hours.
#SBATCH --output=popsplits.%j.out               # sets the standard output to be stored in file my_nice_job.%j.out, where %j is the job id)
#SBATCH --error=popsplits.%j.err                # sets the standard error to be stored in file my_nice_job.%j.err, where % j is the job id)

srun python run_population_splits.py -tp $1 -tf $2
