#!/bin/bash

# run from _scripts folder

for machineVersion in cpu_asus_reflvl6 cpu_bar_reflvl7
do
    for threads in 8 18 36
    do
        testfile=../_results/blockVectorPerm/${machineVersion}/perm_${threads}t_px_py.txt

        if [ ! -f ${testfile} ]
        then
            continue
        fi

        outfile=../_results/blockVectorPerm/${machineVersion}_${threads}t_all.txt

        echo "permx permy V K A D" > ${outfile}

        for x in ox px
        do
            for y in oy py
            do
                echo -n "${x} ${y} " >> ${outfile}

                infile=../_results/blockVectorPerm/${machineVersion}/perm_${threads}t_${x}_${y}.txt

                times="$(cat ${infile} | grep "BESTHEA Info: apply, apply itself elapsed time" | tr -s ' ' | cut -d' ' -f 9)"

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

    done

done


