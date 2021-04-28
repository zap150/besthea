#!/bin/bash

# run from the build directory

quadr_order=4
reflvl=8
threads=24

export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,12c
export OMP_NUM_THREADS=${threads}

for ver in 1 2 3 4
do
    for gpucount in 1 2 3 4
    do
        gpus=0

        for (( g=1; g<${gpucount}; g++ ))
        do
            gpus="${gpus},${g}"
        done

        export CUDA_VISIBLE_DEVICES=${gpus}

        ./experiments/matrix_vector_product ${reflvl} ${ver} 10 2 ${quadr_order} ${quadr_order} > ../experiments/_results/paraScalingStrongGPU/parascale_bar_qo${quadr_order}_reflvl${reflvl}/parascale_ver${ver}_${gpucount}gpus.txt

    done

done



