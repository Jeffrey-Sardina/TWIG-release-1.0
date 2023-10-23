#!/bin/bash

# Move into the rec directory
cd rec_construct_1/

for version in 2
do
    for dataset in UMLS
    do
        n_first_epochs=50
        n_second_epochs=100
        out_file="rec_v${version}_${dataset}_e${n_first_epochs}-e${n_second_epochs}.log"

        echo "Running $out_file"
        exp_start=`date +%s`
        python -u run_exp.py \
            $version \
            $dataset \
            $n_first_epochs \
            $n_second_epochs \
            &> $out_file
        exp_end=`date +%s`
        exp_runtime=$((end-start))
        echo "Experiments took $exp_runtime seconds" 1>> $out_file
    done
done

