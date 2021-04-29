#!/bin/bash

# run from the build directory

quadr_order=4
threads=24
ver=2
ratio=hx_ht

export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,12c
export OMP_NUM_THREADS=${threads}


for reflvl in 2 3 4 5 6 7 8 9
do

    ./experiments/matrix_solve ${reflvl} ${ver} 1 0 ${quadr_order} ${quadr_order} > ../experiments/_results/convergence/convergence_${ratio}/conv_reflvl${reflvl}.txt

done

