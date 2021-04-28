#!/bin/bash

# run from the _scripts directory


measuredThings=(mem_assemble mem_multiply mem_total fly_mult_cpu fly_mult_gpu)
measuredThingFiles=(mem mem mem flycpu flygpu)


for machineVersion in bar_mv_qo4 asus_mv_qo4
do
    machineFolder=../_results/totalComparison/totalcomp_${machineVersion}
    outfile=../_results/totalComparison/totalcomp_${machineVersion}_all.txt
    echo -n > ${outfile}

    for reflvl in 2 3 4 5 6 7 8 9
    do
        echo "reflvl${reflvl}" >> ${outfile}
        echo "measurement V K A D" >> ${outfile}

        for (( i=0; i<5; i++ ))
        do
            measThing=${measuredThings[$i]}
            measFile=${measuredThingFiles[$i]}
            
            infile=${machineFolder}/totalcomp_${measFile}_reflvl${reflvl}.txt

            if [ -f ${infile} ]
            then            
                cat ${infile} | grep "${measThing}" >> ${outfile}
            else
                echo "${measThing} 0 0 0 0" >> ${outfile}
            fi

        done

        echo >> ${outfile}
        echo >> ${outfile}

    done

done

