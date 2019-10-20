#!/bin/bash

echo Start experiments. Good luck!

# copy experiment files
echo copy the source experiment file
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
 for i in `seq 1 8`
 do
  #mkdir ${dirlist}${i}
  cp -r ${dirlist} ${dirlist}-No${i}
 done
done

# training
echo start training
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist
python run_mujoco.py >& result-training.txt &
sleep 5s
cd ../
done

wait

# test
echo start testing
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist
# POLPATH=$(cat bestpolid.txt) # don't use anymore as this way does not make fair comparison 20190128
POLPATH=$(cat latestpolid.txt)
python run_mujoco.py --mpath ${POLPATH} >& result-test.txt &
sleep 5s
cd ../
done

wait

# training
echo start training
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist
python run_mujoco.py >& result-training1.txt &
sleep 5s
cd ../
done

wait

# test
echo start testing
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist
# POLPATH=$(cat bestpolid.txt) # don't use anymore as this way does not make fair comparison 20190128
POLPATH=$(cat latestpolid.txt)
python run_mujoco.py --mpath ${POLPATH} >& result-test1.txt &
sleep 5s
cd ../
done

wait

# training
echo start training
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist
python run_mujoco.py >& result-training2.txt &
sleep 5s
cd ../
done

wait

# test
echo start testing
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist
# POLPATH=$(cat bestpolid.txt) # don't use anymore as this way does not make fair comparison 20190128
POLPATH=$(cat latestpolid.txt)
python run_mujoco.py --mpath ${POLPATH} >& result-test2.txt &
sleep 5s
cd ../
done

wait

# training
echo start training
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist
python run_mujoco.py >& result-training3.txt &
sleep 5s
cd ../
done

wait

# test
echo start testing
for dirlist in `ls -l | awk '$1 ~ /d/ {print $9}'`
do
echo $dirlist
cd $dirlist
# POLPATH=$(cat bestpolid.txt) # don't use anymore as this way does not make fair comparison 20190128
POLPATH=$(cat latestpolid.txt)
python run_mujoco.py --mpath ${POLPATH} >& result-test3.txt &
sleep 5s
cd ../
done
