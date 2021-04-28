#!/bin/bash

# run from the build directory

quadr_order=4
threads=24
ver=2

export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,12c
export OMP_NUM_THREADS=${threads}


for reflvl in 9 8 7 6 5 4 3 2
do

    ./experiments/matrix_vector_product ${reflvl} ${ver} 20 0 ${quadr_order} ${quadr_order} > ../experiments/_results/loadBalancing/loadbalancing_bar_qo${quadr_order}/loadbalancing_reflvl${reflvl}.txt

done


