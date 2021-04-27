#!/bin/bash

# run from the _scripts directory


for machineVersion in bar_qo4_reflvl7 bar_qo1_reflvl7
do

    machineFolder=../_results/paraScalingStrongCPU/parascale_${machineVersion}
    outfile=../_results/paraScalingStrongCPU/parascale_${machineVersion}_all_matrix.txt
    echo "threadcount V K A D" > ${outfile}

    for infile in ${machineFolder}/parascale_*
    do
        filename=$(echo ${infile} | cut -d'/' -f 5)
        threadcount=${filename:10:2}

        echo -ne "${threadcount}\t" >> ${outfile}

        times="$(cat ${infile} | grep "BESTHEA Info: apply, apply itself elapsed time" | tr -s ' ' | cut -d' ' -f 9)"

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

done


