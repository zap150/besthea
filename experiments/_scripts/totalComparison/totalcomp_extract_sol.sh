#!/bin/bash

# run from the _scripts directory


filekinds=(mem flycpu flygpu)
kindgreps=(mem fly_cpu fly_gpu)
bcs=(dir neu)
bccutfields=(3 4)


for machineVersion in bar_sol_qo4 asus_sol_qo4
do
    machineFolder=../_results/totalComparison/totalcomp_${machineVersion}
    outfile=../_results/totalComparison/totalcomp_${machineVersion}_all.txt
    echo -n > ${outfile}

    for reflvl in 2 3 4 5 6 7 8 9
    do
        for bc in 0 1
        do
            echo "reflvl${reflvl} ${bcs[${bc}]}" >> ${outfile}
            echo "implem prepare solve total" >> ${outfile}

            for kind in 0 1 2
            do
                infile=${machineFolder}/totalcomp_${filekinds[${kind}]}_reflvl${reflvl}.txt

                if [ -f ${infile} ]
                then
                    tmsolve=$(cat ${infile} | grep "solve ${kindgreps[${kind}]}" | tail -n 1 | tr -s ' ' | cut -d' ' -f ${bccutfields[${bc}]})
                    tmtotal=$(cat ${infile} | grep "total ${kindgreps[${kind}]}" | tr -s ' ' | cut -d' ' -f ${bccutfields[${bc}]})
                else
                    tmsolve=0
                    tmtotal=0
                fi

                tmprep=`awk "BEGIN{print ${tmtotal}-${tmsolve}}"`

                echo -e "${filekinds[${kind}]}\t${tmprep}\t${tmsolve}\t${tmtotal}" >> ${outfile}


            done

            echo >> ${outfile}

        done




        echo >> ${outfile}
        echo >> ${outfile}

    done

done

