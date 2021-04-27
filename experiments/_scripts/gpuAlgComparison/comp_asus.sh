#!/bin/bash

# run from the build directory

quadr_order=4
threads=8

export KMP_AFFINITY=granularity=core,compact
export OMP_NUM_THREADS=${threads}

for ver in 1 2 3 4
do
    for reflvl in 2 3 4 5 6
    do
        ./experiments/matrix_vector_product ${reflvl} ${ver} 10 2 ${quadr_order} ${quadr_order} > ../experiments/_results/gpuAlgComparison/algcomp_asus_qo${quadr_order}/algcomp_ver${ver}_reflvl${reflvl}.txt
    done

    for reflvl in 7
    do
        ./experiments/matrix_vector_product ${reflvl} ${ver} 3 1 ${quadr_order} ${quadr_order} > ../experiments/_results/gpuAlgComparison/algcomp_asus_qo${quadr_order}/algcomp_ver${ver}_reflvl${reflvl}.txt
    done

done



