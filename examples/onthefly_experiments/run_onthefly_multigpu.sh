#!/bin/bash

warmups=2
repetitions=10
executable=bin/besthea/onthefly_multiply

if [ ! -f ${executable} ]
then
    echo "Please cd to the directory where besthea is installed"
    exit 1
fi

outdir=onthefly_experiments_out/multigpu
mkdir -p ${outdir}
datestr=$(date +%Y%m%d-%H%M%S)
resfile="${outdir}/${datestr}_result.txt"
echo -e "host ${HOSTNAME}\n" > ${resfile}

echo "                 V            K            KT           D" >> ${resfile}

# applies only to the Karolina nodes, change it on other clusters
export OMP_NUM_THREADS=128
export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,64c

for ngpus in {1..8}
do
    echo "ngpus ${ngpus}"
    echo -n "${ngpus} gpus" >> ${resfile}

    outfile="${outdir}/${datestr}_out_${ngpus}.txt"

    COMMAND="${executable}"
    COMMAND+=" --mesh ${PWD}/bin/besthea/cube_24.txt"
    COMMAND+=" --space-refine 4"
    COMMAND+=" --timesteps 256"
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
    CUDA_VISIBLE_DEVICES=$(seq -s"," 0 ${maxgpuidx}) ${COMMAND} >> ${outfile}

    grep "fly_mult_gpu" ${outfile} | sed "s/fly_mult_gpu//" >> ${resfile}

done

