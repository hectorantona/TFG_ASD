#!/bin/bash

#SBATCH --job-name=multiscale_nhpp                # sets the job name
#SBATCH --cpus-per-task=1                         # sets 1 core for each task
#SBATCH --mem=10Gb                               # reserves 100 GB memory
#SBATCH --partition=ioannidis                  # requests that the job is executed in partition my partition
#SBATCH --time=10:00:00                            # reserves machines/cores for 4 hours.
#SBATCH --output=multiscale_nhpp.%j.out               # sets the standard output to be stored in file my_nice_job.%j.out, where %j is the job id)
#SBATCH --error=multiscale_nhpp.%j.err                # sets the standard error to be stored in file my_nice_job.%j.err, where %j is the job id)

srun python simulation_multiscale_NHPP.py