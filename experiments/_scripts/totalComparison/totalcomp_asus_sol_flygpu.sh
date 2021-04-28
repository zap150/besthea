#!/bin/bash

# run from the build directory

quadr_order=4
threads=8
ver=2
type=flygpu

export KMP_AFFINITY=granularity=core,compact
export OMP_NUM_THREADS=${threads}


for reflvl in 2 3 4 5 6
do

    ./experiments/matrix_solve ${reflvl} ${ver} 3 1 ${quadr_order} ${quadr_order} > ../experiments/_results/totalComparison/totalcomp_asus_sol_qo${quadr_order}/totalcomp_${type}_reflvl${reflvl}.txt

done


for reflvl in 7
do

    ./experiments/matrix_solve ${reflvl} ${ver} 1 0 ${quadr_order} ${quadr_order} > ../experiments/_results/totalComparison/totalcomp_asus_sol_qo${quadr_order}/totalcomp_${type}_reflvl${reflvl}.txt

done

