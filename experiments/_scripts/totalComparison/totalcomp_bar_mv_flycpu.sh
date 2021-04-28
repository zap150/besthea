#!/bin/bash

# run from the build directory

quadr_order=4
threads=36
ver=2
type=flycpu

export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,18c
export OMP_NUM_THREADS=${threads}


for reflvl in 2 3 4 5 6 7
do

    ./experiments/matrix_vector_product ${reflvl} ${ver} 10 2 ${quadr_order} ${quadr_order} > ../experiments/_results/totalComparison/totalcomp_bar_mv_qo${quadr_order}/totalcomp_${type}_reflvl${reflvl}.txt

done

for reflvl in 8
do

    ./experiments/matrix_vector_product ${reflvl} ${ver} 3 1 ${quadr_order} ${quadr_order} > ../experiments/_results/totalComparison/totalcomp_bar_mv_qo${quadr_order}/totalcomp_${type}_reflvl${reflvl}.txt

done

