#!/bin/bash

# run from the build directory

reflvl=7
threads=8
x=ox
y=oy


export OMP_NUM_THREADS=${threads}
export KMP_AFFINITY=granularity=core,compact


for gpu_version in 1 2 3 4
do

    ./experiments/matrix_vector_product ${reflvl} ${gpu_version} 10 2 1 1 > ../experiments/_results/blockVectorPerm/gpu_asus_reflvl${reflvl}/perm_ver${gpu_version}_${x}_${y}.txt

done


