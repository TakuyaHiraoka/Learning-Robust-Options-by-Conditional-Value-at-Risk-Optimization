#!/bin/bash

echo Start experiments. Good luck!

I=0
J=0
# test
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist

# find savefiles
J=0
for savedir in `ls | grep 'saves'`
do
    echo ${savedir}
    POLID=($(cat ./${savedir}/bestpol-cvar.txt))
    echo ${POLID}
    if [ ${POLID} = "-1" ]; then
        echo "No feasible policies are learned."
    else
         echo "Peform evaluation"
        SAVEFILE=($(ls ./${savedir}/ | grep epoch_${POLID}. | sed -E 's/.ckpt.*//g'))
        echo ${SAVEFILE[0]}
        python run_mujoco.py --mpath ./${savedir}/${SAVEFILE[0]}.ckpt >& result-test-epoch-${J}.txt &
        I=`expr $I + 1`
        J=`expr $J + 1`

        sleep 5s
    fi

    if [ ${I} -gt 2 ]; then
    echo Wait till all runninng python proccess are finished.
    wait
    #sleep 30m
    I=0
    fi

done

cd ../
done

