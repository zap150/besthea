#!/bin/bash

# run from the _scripts directory


for machineVersion in bar_qo4 asus_qo4
do

    machineFolder=../_results/loadBalancing/loadbalancing_${machineVersion}
    outfile=../_results/loadBalancing/loadbalancing_${machineVersion}_all.txt
    echo -n > ${outfile}

    matIndex=0
    for matrix in V K A D
    do

        for reflvl in 2 3 4 5 6 7 8 9
        do
            infile=${machineFolder}/loadbalancing_reflvl${reflvl}.txt

            if [ ! -f ${infile} ]
            then
                continue
            fi

            echo "${matrix} reflvl${reflvl}" >> ${outfile}
            echo "iter cpuelem timecpu timegpu timetotal" >> ${outfile}

            elemsCpu=$(cat ${infile} | grep "BESTHEA Info: onthefly load balancing: total" | tr -s ' ' | cut -d' ' -f 9)
            timesCpu=$(cat ${infile} | grep "BESTHEA Info: time cpu_all" | tr -s ' ' | cut -d' ' -f 5)
            timesGpu=$(cat ${infile} | grep "BESTHEA Info: time gpu_max" | tr -s ' ' | cut -d' ' -f 5)
            timesTotal=$(cat ${infile} | grep "BESTHEA Info: apply elapsed time" | tr -s ' ' | cut -d' ' -f 7)
            
            timescount=$(echo ${timesCpu} | wc -w)
            repstotal=$((timescount / 4))
            
            matDataStartI=$((repstotal*matIndex))

            for (( iter=0; iter<repstotal; iter++ ))
            do
                iterfrom1=$((iter+1))
                echo -ne "${iterfrom1}\t" >> ${outfile}

                rowidx=$((matDataStartI+iter))

                i=0
                for a in ${elemsCpu}; do
                    if [ ${i} -eq ${rowidx} ]; then
                        echo -ne "${a}\t" >> ${outfile}
                        break
                    fi
                    i=$((i+1))
                done

                i=0
                for a in ${timesCpu}; do
                    if [ ${i} -eq ${rowidx} ]; then
                        echo -ne "${a}\t" >> ${outfile}
                        break
                    fi
                    i=$((i+1))
                done

                i=0
                for a in ${timesGpu}; do
                    if [ ${i} -eq ${rowidx} ]; then
                        echo -ne "${a}\t" >> ${outfile}
                        break
                    fi
                    i=$((i+1))
                done

                i=0
                for a in ${timesTotal}; do
                    if [ ${i} -eq ${rowidx} ]; then
                        echo -ne "${a}\t" >> ${outfile}
                        break
                    fi
                    i=$((i+1))
                done

                echo >> ${outfile}
            
            done

            echo >> ${outfile}
            echo >> ${outfile}

        done




        echo >> ${outfile}
        echo >> ${outfile}
        echo >> ${outfile}
        echo >> ${outfile}
        echo >> ${outfile}
        matIndex=$((matIndex+1))
    done

done


