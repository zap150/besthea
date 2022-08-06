#!/bin/bash

ml intel/2022a
ml KAROLINA/FAKEintel
ml CUDA/11.4.1
export OMP_NUM_THREADS=128
export GOMP_CPU_AFFINITY=0-127
export MKL_NUM_THREADS=64



warmups=2
repetitions=10
executable=bin/besthea/onthefly_multiply

if [ ! -f ${executable} ]
then
    echo "Please cd to the directory where besthea is installed"
    exit 1
fi

outdir=onthefly_experiments_out/comparison_mult
mkdir -p ${outdir}
datestr=$(date +%Y%m%d-%H%M%S)
resfile="${outdir}/${datestr}_result.txt"
echo "host ${HOSTNAME}" > ${resfile}

# finess_level   1  2   3   4   5    6    7    8     9
# n_timesteps    2  4   8  16  32   64  128  256   512
# n_space_elems 48 96 192 384 768 1536 3072 6144 12288  ...
# base_sp_elems 12 24  12  24  12   24   12   24    12
# space_refine   1  1   2   2   3    3    4    4     5

for finess_level in {3..9}
do
    echo "finess_level ${finess_level}"

    outfile="${outdir}/${datestr}_out_${finess_level}.txt"

    timesteps=$(( 2**${finess_level} ))
    space_refine=$(( (${finess_level} + 1) / 2 ))
    mesh_base_elems=$(( (((${finess_level} + 1) % 2) + 1) * 12 ))

    COMMAND="${executable}"
    COMMAND+=" --mesh ${PWD}/bin/besthea/cube_${mesh_base_elems}.txt"
    COMMAND+=" --space-refine ${space_refine}"
    COMMAND+=" --timesteps ${timesteps}"
    COMMAND+=" --endtime 1"
    COMMAND+=" --hc 1"
    if [ ${finess_level} -le 8 ]; then
        COMMAND+=" --do-inmemory"
    fi
    if [ ${finess_level} -le 8 ]; then
        COMMAND+=" --do-onthefly-cpu"
    fi
    COMMAND+=" --do-onthefly-gpu"
    COMMAND+=" --do-V"
    COMMAND+=" --do-K"
    COMMAND+=" --do-KT"
    COMMAND+=" --do-D"
    COMMAND+=" --warmups ${warmups}"
    COMMAND+=" --repetitions ${repetitions}"
    COMMAND+=" --qo-singular 4"
    COMMAND+=" --qo-regular 4"

    ${COMMAND} >> ${outfile}

    echo -e "\nfiness_level ${finess_level}" >> ${resfile}
    tail ${outfile} -n 6 >> ${resfile}

done
