#!/bin/bash

# run from the _scripts directory



for ver in 1 2
do
    for machineVersion in tpb_bar_reflvl8 tpb_asus_reflvl6
    do
        outfile=../_results/threadblockDim/${machineVersion}_ver${ver}_all.txt
        echo -n "" > ${outfile}

        for qo in 1 2 4 5
        do
            echo "quadr order ${qo}" >> ${outfile}
            echo "tpb V K A D" >> ${outfile}

            for infile in ../_results/threadblockDim/${machineVersion}/tpb_ver${ver}_qo${qo}_*
            do
                tpb="$(echo ${infile} | cut -d'_' -f 7 | cut -d'.' -f 1)"
                echo -ne "${tpb}\t" >> ${outfile}

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
                        echo -ne "${avg}  \t" >> ${outfile}
                    fi

                    index=$((index+1))
                    
                done

                echo >> ${outfile}

            done

            echo >> ${outfile}
            echo >> ${outfile}

        done

    done

done


