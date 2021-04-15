#!/bin/bash

# run from the build directory

reflvl=7
threads=18
x=px
y=py


export OMP_NUM_THREADS=${threads}
export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=1s,18c



./experiments/matrix_vector_product ${reflvl} 1 10 2 1 1 > ../experiments/_results/blockVectorPerm/cpu_bar_reflvl${reflvl}/perm_${threads}t_${x}_${y}.txt



