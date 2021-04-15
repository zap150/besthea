#!/bin/bash

# run from _scripts folder


for ver in v1 v2 v3 v4
do
    outfile=../_results/blockVectorPerm/perm_gpu_${ver}_all.txt
    echo "permx permy V K A D" > ${outfile}

    for x in ox px
    do
        for y in oy py
        do
            echo -n "${x} ${y} " >> ${outfile}

            infile=../_results/blockVectorPerm/perm_gpu_${ver}_${x}_${y}.txt

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

done


