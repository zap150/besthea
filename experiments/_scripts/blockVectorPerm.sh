#!/bin/bash

# run from the build directory

refine=2
threads=36



export OMP_NUM_THREADS=${threads}
export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,18c



./experiments/matrix_vector_product ${refine} 1 10 2 1 1


