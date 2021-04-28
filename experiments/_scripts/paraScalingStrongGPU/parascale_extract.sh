#!/bin/bash

# run from the _scripts directory


for machineVersion in bar_qo4_reflvl8 bar_qo4_reflvl9
do

    machineFolder=../_results/paraScalingStrongGPU/parascale_${machineVersion}
    outfile=../_results/paraScalingStrongGPU/parascale_${machineVersion}_all.txt
    echo > ${outfile}

    for version in 1 2 3 4
    do
        echo "ver${version}" >> ${outfile}
        echo "gpucount V K A D" >> ${outfile}

        for gpucount in 1 2 3 4
        do
            echo -ne "${gpucount}\t" >> ${outfile}

            infile=${machineFolder}/parascale_ver${version}_${gpucount}gpus.txt

            times="$(cat ${infile} | grep "BESTHEA Info: time gpu_max" | tr -s ' ' | cut -d' ' -f 5)"

            timescount=$(echo ${times} | wc -w)
            repstotal=$((${timescount} / 4))
            repstotalminusone=$((${repstotal} - 1))

            if [ ${repstotal} -eq 1 ]; then
                prereps=0
                reps=1
            elif [ ${repstotal} -eq 4 ]; then
                prereps=1
                reps=3
            else
                prereps=2
                reps=10
            fi


            index=0
            total=0
            for a in ${times}
            do
                m=$((index % ${repstotal}))

                if [ "${m}" -eq "0" ]
                then
                    total=0
                fi

                if [ "${m}" -ge "${prereps}" ]
                then
                    total=`awk "BEGIN{print ${total}+${a}}"`
                fi
                
                if [ "${m}" -eq "${repstotalminusone}" ]
                then
                    avg=`awk "BEGIN{print ${total}/${reps}.0}"`
                    echo -n "${avg} " >> ${outfile}
                fi

                index=$((index+1))
                
            done

            echo >> ${outfile}


        done

        echo >> ${outfile}
        echo >> ${outfile}


    done


done


