#!/bin/bash

# run from the build directory

reflvl=6
threads=8
x=px
y=oy


export OMP_NUM_THREADS=${threads}
export KMP_AFFINITY=granularity=core,compact



./experiments/matrix_vector_product ${reflvl} 1 10 2 1 1 > ../experiments/_results/blockVectorPerm/cpu_asus_reflvl${reflvl}/perm_${threads}t_${x}_${y}.txt



