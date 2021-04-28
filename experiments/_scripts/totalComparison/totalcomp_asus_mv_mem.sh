#!/bin/bash

# run from the build directory

quadr_order=4
threads=8
ver=2
type=mem

export KMP_AFFINITY=granularity=core,compact
export OMP_NUM_THREADS=${threads}


for reflvl in 6
do

    ./experiments/matrix_vector_product ${reflvl} ${ver} 3 1 ${quadr_order} ${quadr_order} > ../experiments/_results/totalComparison/totalcomp_asus_mv_qo${quadr_order}/totalcomp_${type}_reflvl${reflvl}.txt

done

for reflvl in 5 4 3 2
do

    ./experiments/matrix_vector_product ${reflvl} ${ver} 10 2 ${quadr_order} ${quadr_order} > ../experiments/_results/totalComparison/totalcomp_asus_mv_qo${quadr_order}/totalcomp_${type}_reflvl${reflvl}.txt

done

