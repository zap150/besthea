#!/bin/bash

# run from the _scripts directory


for machineVersion in bar_qo4 bar_qo1
do

    machineFolder=../_results/componentRatios/comp_${machineVersion}
    outfile=../_results/componentRatios/comp_${machineVersion}_all.txt
    echo -n > ${outfile}

    matIndex=0
    for matrix in V K A D
    do
        echo "${matrix} FR TRSS TS" >> ${outfile}
        matDataStartI=$((12*matIndex))

        for infile in ${machineFolder}/comp_reflvl*
        do
            filename=$(echo ${infile} | cut -d'/' -f 5)
            refinement=${filename:11:1}

            times=$(cat ${infile} | grep "fully-regular\|time-regular-space-singular\|time-singular" | tr -s ' ' | cut -d' ' -f 8)
            


            index=0
            totalFR=0
            totalTRSS=0
            totalTS=0
            for a in ${times}
            do
                c=$((index % 3))
                i=$((index / 3))
                m=$((i % 12))

                if [ ${i} -ge ${matDataStartI} ]
                then

                    if [ "${m}" -ge "2" ]
                    then
                        if [ ${c} -eq 0 ]; then
                            totalFR=`awk "BEGIN{print ${totalFR}+${a}}"`
                        elif [ ${c} -eq 1 ]; then
                            totalTRSS=`awk "BEGIN{print ${totalTRSS}+${a}}"`
                        else
                            totalTS=`awk "BEGIN{print ${totalTS}+${a}}"`
                        fi
                    fi

                    if [ "${m}" -eq "11" ] && [ ${c} -eq 2 ]
                    then
                        avgFR=`awk "BEGIN{print ${totalFR}/10.0}"`
                        avgTRSS=`awk "BEGIN{print ${totalTRSS}/10.0}"`
                        avgTS=`awk "BEGIN{print ${totalTS}/10.0}"`

                        echo -ne "ref${refinement}\t" >> ${outfile}
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


