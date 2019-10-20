#!/bin/bash

echo Start experiments. Good luck!

EPOCH=430
I=0
J=0
# test
echo start testing at ${EPOCH}th epoch
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist

# find savefiles
J=0
for savedir in `ls | grep 'saves'`
do
echo ${savedir}
SAVEFILE=($(ls ./${savedir}/ | grep -C 1 epoch_${EPOCH} | sed -E 's/.ckpt.*//g'))
#echo ${SAVEFILE}
echo ${SAVEFILE[0]}
python run_mujoco.py --mpath ./${savedir}/${SAVEFILE[0]}.ckpt >& result-test-epoch${EPOCH}-${J}.txt &
I=`expr $I + 1`
J=`expr $J + 1`

sleep 5s

if [ ${I} -gt 8 ]; then
echo Wait till all runninng python proccess are finished.
wait
#sleep 30m
I=0
fi

done

cd ../
done

