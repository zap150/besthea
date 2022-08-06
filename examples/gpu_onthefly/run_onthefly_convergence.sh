#!/bin/bash

executable=bin/besthea/onthefly_solve

if [ ! -f ${executable} ]
then
    echo "Please cd to the directory where besthea is installed"
    exit 1
fi

outdir=onthefly_experiments_out/convergence
mkdir -p ${outdir}
datestr=$(date +%Y%m%d-%H%M%S)

# applies to the Karolina nodes, change it on other clusters
export OMP_NUM_THREADS=128
export KMP_AFFINITY=granularity=core,compact
export KMP_HW_SUBSET=2s,64c

resfile="${outdir}/${datestr}_result.txt"
echo "host ${HOSTNAME}" > ${resfile}

echo -e "\nhx^2 ~=~ ht" >> ${resfile}
echo "\nfiness_level timesteps space_elements rel_error_dir rel_error_neu" >> ${resfile}
# hx^2 ~=~ ht
#
# finess_level   1  2   3   4   5    6    7    8     9
# n_timesteps    2  4   8  16  32   64  128  256   512
# n_space_elems 48 96 192 384 768 1536 3072 6144 12288  ...
# base_sp_elems 12 24  12  24  12   24   12   24    12
# space_refine   1  1   2   2   3    3    4    4     5
for finess_level in {3..9}
do
    echo "finess_level ${finess_level}"

    outfile="${outdir}/${datestr}_out_q${finess_level}.txt"

    timesteps=$(( 2**${finess_level} ))
    space_refine=$(( (${finess_level} + 1) / 2 ))
    mesh_base_elems=$(( (((${finess_level} + 1) % 2) + 1) * 12 ))

    COMMAND="${executable}"
    COMMAND+=" --mesh ${PWD}/bin/besthea/cube_${mesh_base_elems}.txt"
    COMMAND+=" --space-refine ${space_refine}"
    COMMAND+=" --timesteps ${timesteps}"
    COMMAND+=" --endtime 1"
    COMMAND+=" --fgmres_prec 1e-8"
    COMMAND+=" --hc 0.5"
    COMMAND+=" --do-onthefly-gpu"
    COMMAND+=" --do-dirichlet"
    COMMAND+=" --do-neumann"
    COMMAND+=" --warmups 0"
    COMMAND+=" --repetitions 1"
    COMMAND+=" --qo-singular 4"
    COMMAND+=" --qo-regular 4"

    ${COMMAND} >> ${outfile}

    direrr=$(grep "Dir rel_error:" ${outfile} | tr -s ' ' | cut -d' ' -f5)
    neuerr=$(grep "Neu rel_error:" ${outfile} | tr -s ' ' | cut -d' ' -f5)
    echo ${finess_level} ${timesteps} $(( 12*(2**(finess_level+1) ) )) ${direrr} ${neuerr} >> ${resfile}
    
done



echo -e "\nhx ~=~ ht" >> ${resfile}
echo "\nfiness_level timesteps space_elements rel_error_dir rel_error_neu" >> ${resfile}
# hx ~=~ ht
#
# finess_level   1  2   3   4    5     6
# n_timesteps    2  4   8  16   32    64
# n_space_elems 12 48 192 768 3072 12288  ...
# base_sp_elems 12 12  12  12   12    12
# space_refine   0  1   2   3    4     5
for finess_level in {3..6}
do
    echo "finess_level ${finess_level}"

    outfile="${outdir}/${datestr}_out_l${finess_level}.txt"

    timesteps=$(( 2**${finess_level} ))
    space_refine=$(( ${finess_level} - 1 ))
    mesh_base_elems=12

    COMMAND="${executable}"
    COMMAND+=" --mesh ${PWD}/bin/besthea/cube_${mesh_base_elems}.txt"
    COMMAND+=" --space-refine ${space_refine}"
    COMMAND+=" --timesteps ${timesteps}"
    COMMAND+=" --endtime 1"
    COMMAND+=" --fgmres_prec 1e-8"
    COMMAND+=" --hc 0.5"
    COMMAND+=" --do-onthefly-gpu"
    COMMAND+=" --do-dirichlet"
    COMMAND+=" --do-neumann"
    COMMAND+=" --warmups 0"
    COMMAND+=" --repetitions 1"
    COMMAND+=" --qo-singular 4"
    COMMAND+=" --qo-regular 4"

    ${COMMAND} >> ${outfile}

    direrr=$(grep "Dir rel_error:" ${outfile} | tr -s ' ' | cut -d' ' -f5)
    neuerr=$(grep "Neu rel_error:" ${outfile} | tr -s ' ' | cut -d' ' -f5)
    echo ${finess_level} ${timesteps} $(( 12*(4**(finess_level-1) ) )) ${direrr} ${neuerr} >> ${resfile}

done
