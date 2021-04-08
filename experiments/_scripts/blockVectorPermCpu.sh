#!/bin/bash

# run from the build directory

tempRefineStart=1
tempRefineEnd=3
spatRefineStart=1
spatRefineEnd=4
threads=18



export OMP_NUM_THREADS=${threads}
export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=1s,${threads}c

for (( tempRef=${tempRefineStart}; tempRef<=${tempRefineEnd}; tempRef++ ))
do

    for (( spatRef=${spatRefineStart}; spatRef<=${spatRefineEnd}; spatRef++ ))
    do

        time="$(../build/experiments/matrix_vector_product ${tempRef} ${spatRef} 1 1 0 | grep fly_mult_cpu | tr -s " " | cut -d' ' -f 2)"

        echo "tempRef ${tempRef} spatRef ${spatRef} time ${time}"

    done

done

