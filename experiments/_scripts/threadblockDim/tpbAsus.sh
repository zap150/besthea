#!/bin/bash

# run from the build directory

reflvl=6
threads=8

tpb=01x32


export OMP_NUM_THREADS=${threads}
export KMP_AFFINITY=granularity=core,compact


for gpu_version in 3 4
do
    for quadr_order_reg in 1 2 4 5
    do

        ./experiments/matrix_vector_product ${reflvl} ${gpu_version} 3 1 ${quadr_order_reg} 1 > ../experiments/_results/threadblockDim/tpb_asus_reflvl${reflvl}/tpb_ver${gpu_version}_qo${quadr_order_reg}_${tpb}.txt

    done
done



