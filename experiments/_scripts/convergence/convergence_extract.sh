#!/bin/bash

# run from the _scripts directory


bcs=(Dir Neu)
timecutshifts=(3 4)


for ratio in hx2_ht hx_ht
do
    infolder=../_results/convergence/convergence_${ratio}
    outfile=../_results/convergence/convergence_${ratio}.txt    
    echo -n > ${outfile}

    for bc in 0 1
    do
        currbc=${bcs[${bc}]}
        echo ${currbc} >> ${outfile}
        echo "reflvl iters rel_error time" >> ${outfile}

        for infile in ${infolder}/conv_reflvl*
        do
            filename=$(echo ${infile} | cut -d'/' -f 5)
            reflvl=${filename:11:1}

            iters=$(cat ${infile} | grep "${currbc} iterations" | tr -s ' ' | cut -d' ' -f 5)
            relerror=$(cat ${infile} | grep "${currbc} rel_error" | tr -s ' ' | cut -d' ' -f 5)
            tm=$(cat ${infile} | grep "total fly_gpu" | tr -s ' ' | cut -d' ' -f ${timecutshifts[${bc}]})

            echo -e "${reflvl}\t${iters}\t${relerror}\t${tm}" >> ${outfile}



        done

        echo >> ${outfile}
        echo >> ${outfile}
    done

done


