#!/bin/bash

ml intel/2022a
ml KAROLINA/FAKEintel
ml CUDA/11.4.1
export OMP_NUM_THREADS=128
export GOMP_CPU_AFFINITY=0-127
export MKL_NUM_THREADS=128



executable=bin/besthea/onthefly_multiply

if [ ! -f "${executable}" ]
then
    echo "Please cd to the directory where besthea is installed"
    exit 1
fi

datestr="$(date +%Y%m%d_%H%M%S)"
expdir="gpu_onthefly_experiments_out/gpu_algorithm"
casedir="${expdir}/${datestr}"
outdir="${casedir}/out"
mkdir -p "${outdir}"
ln -s -f -n "${datestr}" "${expdir}/last"
resfile="${casedir}/results.txt"
echo "host ${HOSTNAME}" > "${resfile}"
date >> "${resfile}"

# finess_level   1  2   3   4   5    6    7    8     9
# n_timesteps    2  4   8  16  32   64  128  256   512
# n_space_elems 48 96 192 384 768 1536 3072 6144 12288  ...
# base_sp_elems 12 24  12  24  12   24   12   24    12
# space_refine   1  1   2   2   3    3    4    4     5

finess_level_start=3
finess_level_stop=10

for matrix in V K KT D
do
    for finess_level in $(seq "${finess_level_start}" "${finess_level_stop}")
    do
        timesteps=$(( 2**${finess_level} ))
        space_refine=$(( (${finess_level} + 1) / 2 ))
        mesh_base_elems=$(( (((${finess_level} + 1) % 2) + 1) * 12 ))

        warmups=0
        repetitions=0
        if [[ "${finess_level}" -le 9 ]]; then
            warmups=1
            repetitions=10
        elif [[ "${finess_level}" -eq 10 ]]; then
            warmups=1
            repetitions=3
        fi

        for gpu_alg in 1 2 3 4
        do

            outfile="${outdir}/out_${matrix}_${gpu_alg}_${finess_level}.txt"

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
            COMMAND+=" --gpu-alg ${gpu_alg}"

            ${COMMAND} >> "${outfile}"

        done
    done
done


for matrix in V K KT D
do
    echo -e "\n${matrix}" >> "${resfile}"
    echo -n "finess " >> "${resfile}"
    seq -s " " "${finess_level_start}" "${finess_level_stop}" >> "${resfile}"

    for gpu_alg in 1 2 3 4
    do
        echo -n "gpu_alg${gpu_alg}" >> "${resfile}"

        for finess_level in $(seq "${finess_level_start}" "${finess_level_stop}")
        do
            if [[ "${matrix}" == V ]]; then
                field=2
            elif [[ "${matrix}" == K ]]; then
                field=3
            elif [[ "${matrix}" == KT ]]; then
                field=4
            elif [[ "${matrix}" == D ]]; then
                field=5
            fi
            
            infile="${outdir}/out_${matrix}_${gpu_alg}_${finess_level}.txt"

            echo -n " " >> "${resfile}"
            tail "${infile}" -n 6 | grep "fly_mult_gpu" | tr -s ' ' | cut -d ' ' -f "${field}" | tr -d '\n' >> "${resfile}"
        done

        echo >> "${resfile}"
    done
done
