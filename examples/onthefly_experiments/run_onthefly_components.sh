#!/bin/bash

warmups=2
repetitions=10
executable=bin/besthea/onthefly_multiply

if [ ! -f ${executable} ]
then
    echo "Please cd to the directory where besthea is installed"
    exit 1
fi

outdir=onthefly_experiments_out/components
mkdir -p ${outdir}
datestr=$(date +%Y%m%d-%H%M%S)
resfile="${outdir}/${datestr}_result.txt"
echo "host ${HOSTNAME}" > ${resfile}

# finess_level   1  2   3   4   5    6    7    8     9
# n_timesteps    2  4   8  16  32   64  128  256   512
# n_space_elems 48 96 192 384 768 1536 3072 6144 12288  ...
# base_sp_elems 12 24  12  24  12   24   12   24    12
# space_refine   1  1   2   2   3    3    4    4     5

# applies to the Karolina nodes, change it on other clusters
export OMP_NUM_THREADS=128
export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,64c

for matrix in V K KT D
do
    echo "matrix ${matrix}"
    echo -e "\nmatrix ${matrix}" >> ${resfile}
    echo "finess_level cpu_delta0 cpu_singular gpu_max" >> ${resfile}

    for finess_level in {3..9}
    do
        echo "  finess_level ${finess_level}"
        echo -n "${finess_level}" >> ${resfile}

        outfile="${outdir}/${datestr}_out_${finess_level}_${matrix}.txt"

        timesteps=$(( 2**${finess_level} ))
        space_refine=$(( (${finess_level} + 1) / 2 ))
        mesh_base_elems=$(( (((${finess_level} + 1) % 2) + 1) * 12 ))

        COMMAND="${executable}"
        COMMAND+=" --mesh ${PWD}/bin/besthea/cube_${mesh_base_elems}.txt"
        COMMAND+=" --space-refine ${space_refine}"
        COMMAND+=" --timesteps ${timesteps}"
        COMMAND+=" --endtime 1"
        COMMAND+=" --hc 1"
        COMMAND+=" --do-onthefly-gpu"
        COMMAND+=" --do-${matrix}"
        COMMAND+=" --warmups ${warmups}"
        COMMAND+=" --repetitions ${repetitions}"
        COMMAND+=" --qo-singular 4"
        COMMAND+=" --qo-regular 4"

        ${COMMAND} > ${outfile}

        for component in cpu_delta0 cpu_singular gpu_max
        do
            times=$(grep "BESTHEA Info: time ${component}:" ${outfile} | tr -s ' ' | cut -d' ' -f5 | tail -n ${repetitions})
            total=0
            for t in ${times}
            do
                total=$(awk "BEGIN{print ${total}+${t}}")
            done
            avg=$(awk "BEGIN{print ${total}/${repetitions}.0}")
            echo -n " ${avg}" >> ${resfile}
        done

        echo >> ${resfile}

    done

done
