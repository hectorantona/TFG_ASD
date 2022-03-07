#!/bin/bash
#for tp in "4" "8" "12"
for tp in "12"
do
#    for tf in "4" "8" "12"
    for tf in "4"
    do
        for nb in $(seq 1 10);
        do
            work=/scratch/talesim/AggressionASD/mcsvm
            cd $work
            sbatch run_batch_popsplit_discovery.sh $tp $tf
        done
    done
done