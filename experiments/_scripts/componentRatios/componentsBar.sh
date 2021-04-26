#!/bin/bash

# run from the build directory

quadr_order=4


export OMP_NUM_THREADS=1
export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,18c


for reflvl in 7 6 5 4 3 2
do

    ./experiments/matrix_vector_product ${reflvl} 1 10 2 ${quadr_order} ${quadr_order} > ../experiments/_results/componentRatios/comp_bar_qo${quadr_order}/comp_reflvl${reflvl}.txt

done



