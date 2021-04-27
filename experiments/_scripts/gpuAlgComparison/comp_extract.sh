#!/bin/bash

# run from the _scripts directory


for machineVersion in bar_qo4 bar_qo1 asus_qo4 asus_qo1
do

    machineFolder=../_results/gpuAlgComparison/algcomp_${machineVersion}
    outfile=../_results/gpuAlgComparison/algcomp_${machineVersion}_all.txt
    echo -n > ${outfile}

    matIndex=0
    for matrix in V K A D
    do
        echo "${matrix}" >> ${outfile}
        echo "version\reflvl 2 3 4 5 6 7 8 9" >> ${outfile}

        for ver in 1 2 3 4
        do
            echo -ne "ver${ver}\t" >> ${outfile}

            for reflvl in 2 3 4 5 6 7 8 9
            do
                infile=${machineFolder}/algcomp_ver${ver}_reflvl${reflvl}.txt

                if [ ! -f ${infile} ]
                then
                    continue
                fi

                times=$(cat ${infile} | grep "BESTHEA Info: time gpu_max" | tr -s ' ' | cut -d' ' -f 5)
                
                timescount=$(echo ${times} | wc -w)
                repstotal=$((timescount / 4))
                repstotalminusone=$((repstotal - 1))
                
                matDataStartI=$((repstotal*matIndex))

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
                    if [ ${index} -ge ${matDataStartI} ]
                    then
                        m=$((index % ${repstotal}))

                        if [ "${m}" -ge "${prereps}" ]
                        then
                            total=`awk "BEGIN{print ${total}+${a}}"`
                        fi
                        
                        if [ "${m}" -eq "${repstotalminusone}" ]
                        then
                            avg=`awk "BEGIN{print ${total}/${reps}.0}"`
                            echo -n "${avg} " >> ${outfile}
                            break
                        fi
                    fi

                    index=$((index+1))
                    
                done


            done

            echo >> ${outfile}

        done




        echo >> ${outfile}
        echo >> ${outfile}
        matIndex=$((matIndex+1))
    done

done


