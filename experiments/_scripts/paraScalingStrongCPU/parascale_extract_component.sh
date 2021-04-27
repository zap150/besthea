#!/bin/bash

# run from the _scripts directory


for machineVersion in bar_qo4_reflvl7 bar_qo1_reflvl7
do

    machineFolder=../_results/paraScalingStrongCPU/parascale_${machineVersion}
    outfile=../_results/paraScalingStrongCPU/parascale_${machineVersion}_all_component.txt
    echo -n > ${outfile}

    matIndex=0
    for matrix in V K A D
    do
        echo "${matrix}" >> ${outfile}
        echo "threadcount FR TRSS TS" >> ${outfile}

        for infile in ${machineFolder}/parascale_*
        do
            filename=$(echo ${infile} | cut -d'/' -f 5)
            threadcount=${filename:10:2}

            times=$(cat ${infile} | grep "fully-regular\|time-regular-space-singular\|time-singular" | tr -s ' ' | cut -d' ' -f 8)

            timescount=$(echo ${times} | wc -w)
            repstotal=$((${timescount} / 12))
            repstotalminusone=$((${repstotal} - 1))
            
            matDataStartI=$((${repstotal}*matIndex))

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
            totalFR=0
            totalTRSS=0
            totalTS=0
            for a in ${times}
            do
                c=$((index % 3))
                i=$((index / 3))
                m=$((i % ${repstotal}))

                if [ ${i} -ge ${matDataStartI} ]
                then

                    if [ "${m}" -ge "${prereps}" ]
                    then
                        if [ ${c} -eq 0 ]; then
                            totalFR=`awk "BEGIN{print ${totalFR}+${a}}"`
                        elif [ ${c} -eq 1 ]; then
                            totalTRSS=`awk "BEGIN{print ${totalTRSS}+${a}}"`
                        else
                            totalTS=`awk "BEGIN{print ${totalTS}+${a}}"`
                        fi
                    fi

                    if [ "${m}" -eq "${repstotalminusone}" ] && [ ${c} -eq 2 ]
                    then
                        avgFR=`awk "BEGIN{print ${totalFR}/${reps}.0}"`
                        avgTRSS=`awk "BEGIN{print ${totalTRSS}/${reps}.0}"`
                        avgTS=`awk "BEGIN{print ${totalTS}/${reps}.0}"`

                        echo -ne "${threadcount}\t" >> ${outfile}
                        echo -ne "${avgFR}  \t" >> ${outfile}
                        echo -ne "${avgTRSS}  \t" >> ${outfile}
                        echo -ne "${avgTS}  \t" >> ${outfile}
                        echo >> ${outfile}

                        break
                    fi

                fi

                index=$((index+1))
                
            done

        done

        echo >> ${outfile}
        echo >> ${outfile}
        echo >> ${outfile}
        matIndex=$((matIndex+1))

    done

done


