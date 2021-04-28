#!/bin/bash

# run from the build directory

quadr_order=4
threads=24
ver=2
type=flygpu

export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,12c
export OMP_NUM_THREADS=${threads}


for reflvl in 9
do

    ./experiments/matrix_vector_product ${reflvl} ${ver} 3 1 ${quadr_order} ${quadr_order} > ../experiments/_results/totalComparison/totalcomp_bar_mv_qo${quadr_order}/totalcomp_${type}_reflvl${reflvl}.txt

done

for reflvl in 8 7 6 5 4 3 2
do

    ./experiments/matrix_vector_product ${reflvl} ${ver} 10 2 ${quadr_order} ${quadr_order} > ../experiments/_results/totalComparison/totalcomp_bar_mv_qo${quadr_order}/totalcomp_${type}_reflvl${reflvl}.txt

done

