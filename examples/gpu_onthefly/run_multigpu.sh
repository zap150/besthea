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
expdir="gpu_onthefly_experiments_out/multigpu"
casedir="${expdir}/${datestr}"
outdir="${casedir}/out"
mkdir -p "${outdir}"
ln -s -f -n "${datestr}" "${expdir}/last"
resfile="${casedir}/results.txt"
echo "host ${HOSTNAME}" > "${resfile}"
date >> "${resfile}"

finess_level_start=3
finess_level_stop=10

for finess_level in $(seq "${finess_level_start}" "${finess_level_stop}")
do
    timesteps=$(( 2**${finess_level} ))
    space_refine=$(( (${finess_level} + 1) / 2 ))
    mesh_base_elems=$(( (((${finess_level} + 1) % 2) + 1) * 12 ))

    for ngpus in {1..8}
    do
        warmups=0
        repetitions=0
        if [[ "${finess_level}" -le 9 ]]; then
            warmups=1
            repetitions=10
        elif [[ "${finess_level}" -eq 10 ]]; then
            warmups=1
            repetitions=3
        fi

        outfile="${outdir}/out_${finess_level}_${ngpus}.txt"

        COMMAND="${executable}"
        COMMAND+=" --mesh ${PWD}/bin/besthea/cube_${mesh_base_elems}.txt"
        COMMAND+=" --space-refine ${space_refine}"
        COMMAND+=" --timesteps ${timesteps}"
        COMMAND+=" --endtime 1"
        COMMAND+=" --hc 1"
        COMMAND+=" --do-onthefly-gpu"
        COMMAND+=" --do-V"
        COMMAND+=" --do-K"
        COMMAND+=" --do-KT"
        COMMAND+=" --do-D"
        COMMAND+=" --warmups ${warmups}"
        COMMAND+=" --repetitions ${repetitions}"
        COMMAND+=" --qo-singular 4"
        COMMAND+=" --qo-regular 4"

        maxgpuidx=$((${ngpus}-1))
        CUDA_VISIBLE_DEVICES=$(seq -s"," 0 ${maxgpuidx}) ${COMMAND} >> "${outfile}"

    done
done



for finess_level in $(seq "${finess_level_start}" "${finess_level_stop}")
do
    echo -e "\nfiness_level ${finess_level}" >> "${resfile}"
    echo "N gpus             V            K            KT           D" >> "${resfile}"

    for ngpus in {1..8}
    do
        echo -n "${ngpus} gpus" >> "${resfile}"

        infile="${outdir}/out_${finess_level}_${ngpus}.txt"

        grep "fly_mult_gpu" "${infile}" | sed "s/fly_mult_gpu//" >> "${resfile}"
    done
done
