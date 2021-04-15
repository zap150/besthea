#!/bin/bash

# run from _scripts folder

for machineVersion in gpu_asus_reflvl7 gpu_bar_reflvl8
do
    folder=../_results/blockVectorPerm/${machineVersion}
    outfile=../_results/blockVectorPerm/${machineVersion}_all.txt
    
    echo -n "" > ${outfile}

    for ver in 1 2 3 4
    do
        echo "version ${ver}" >> ${outfile}
        echo "permx permy V K A D" >> ${outfile}

        for x in ox px
        do
            for y in oy py
            do
                echo -n "${x} ${y} " >> ${outfile}

                infile=${folder}/perm_ver${ver}_${x}_${y}.txt

                times="$(cat ${infile} | grep "BESTHEA Info: time gpu_max" | tr -s ' ' | cut -d' ' -f 5)"

                index=0
                total=0
                for a in ${times}
                do
                    m=$((index % 12))

                    if [ "${m}" -eq "0" ]
                    then
                        total=0
                    fi

                    if [ "${m}" -ge "2" ]
                    then
                        total=`awk "BEGIN{print ${total}+${a}}"`
                    fi
                    
                    if [ "${m}" -eq "11" ]
                    then
                        avg=`awk "BEGIN{print ${total}/10.0}"`
                        echo -n "${avg} " >> ${outfile}
                    fi

                    index=$((index+1))
                    
                done

                echo >> ${outfile}
            done
        done

        echo >> ${outfile}
        echo >> ${outfile}

    done

done


