#!/bin/bash

# run from the build directory

quadr_order=4
reflvl=7

export KMP_AFFINITY=granularity=core,compact


export KMP_HW_SUBSET=1s,18c
for threads in 1 2 3 4
do
    export OMP_NUM_THREADS=${threads}
    threadsStr=$(printf "%02d" ${threads})

    ./experiments/matrix_vector_product ${reflvl} 1 1 0 ${quadr_order} ${quadr_order} > ../experiments/_results/paraScalingStrongCPU/parascale_bar_qo${quadr_order}_reflvl${reflvl}/parascale_${threadsStr}t.txt

done



export KMP_HW_SUBSET=1s,18c
for threads in 6 8 10 12 15 18
do
    export OMP_NUM_THREADS=${threads}
    threadsStr=$(printf "%02d" ${threads})

    ./experiments/matrix_vector_product ${reflvl} 1 3 1 ${quadr_order} ${quadr_order} > ../experiments/_results/paraScalingStrongCPU/parascale_bar_qo${quadr_order}_reflvl${reflvl}/parascale_${threadsStr}t.txt

done


export KMP_HW_SUBSET=2s,18c
for threads in 24 30 36
do
    export OMP_NUM_THREADS=${threads}
    threadsStr=$(printf "%02d" ${threads})

    ./experiments/matrix_vector_product ${reflvl} 1 10 2 ${quadr_order} ${quadr_order} > ../experiments/_results/paraScalingStrongCPU/parascale_bar_qo${quadr_order}_reflvl${reflvl}/parascale_${threadsStr}t.txt

done



