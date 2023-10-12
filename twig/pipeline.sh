#!/bin/bash

exp_name=$1
run_nums=$2
num_processes=$3
# num_processes should be a factor of 1215 for even splitting
# 3 and 5 were the settings we used

start=`date +%s`
seed=None

echo "pipeline configured with $num_processes processes"
echo "if memory issues occur, please restart with fewer processes"

for dataset in UMLS
do
    for run_num in $run_nums
    do
        run_name="$dataset-$exp_name-run2.$run_num"
        echo "running $run_name"

        mkdir output/$run_name &> /dev/null
        python pipeline.py \
            output/$run_name/$run_name.grid \
            output/$run_name/ \
            $num_processes \
            $dataset \
            $seed \
            1> output/$run_name/$run_name.res \
            2> output/$run_name/$run_name.log

        end=`date +%s`
        runtime=$((end-start))
        echo "Experiments took $runtime seconds" 1>> output/$run_name/$run_name.log
        echo "I have just finished run $run_name"
    done
done
