#!/bin/bash
for tp in "4" "8" "12"
#for tp in "12"
do
    for tf in "4" "8" "12"
    #for tf in "4"
    do
        work=/scratch/talesim/AggressionInASD/talesCode_newdata_multiclass
        cd $work
        sbatch run_batch_discovery.sh $tp $tf
    done
done