#!/bin/bash

# run from the build directory

quadr_order=4
threads=8
ver=2

export KMP_AFFINITY=granularity=core,compact
export OMP_NUM_THREADS=${threads}


for reflvl in 7 6 5 4 3 2
do

    ./experiments/matrix_vector_product ${reflvl} ${ver} 20 0 ${quadr_order} ${quadr_order} > ../experiments/_results/loadBalancing/loadbalancing_asus_qo${quadr_order}/loadbalancing_reflvl${reflvl}.txt

done


