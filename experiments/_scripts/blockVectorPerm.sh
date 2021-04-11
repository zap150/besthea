#!/bin/bash

# run from the build directory

refine=3
threads=24
#threads=36
gpu_version=4


export OMP_NUM_THREADS=${threads}
export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,12c
#export KMP_HW_SUBSET=2s,18c



./experiments/matrix_vector_product ${refine} ${gpu_version} 10 2 1 1


