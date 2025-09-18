#!/usr/bin/bash 
#config: --utf-8
##!/bin/pyenv python

task_name="Flat-Lus2"
device="cuda:0"
nohup=false
load_run=""
resume=""
while getopts "t:d:l:rn" arg
do 
case $arg in 
t)
task_name=$OPTARG
;;
d)
device=$OPTARG
;;
r)
resume="--resume=True"
;;
n)
nohup=true
;;
l)
load_run="--load_run $OPTARG"
;;
?)
echo "Unknow args"
;;
esac
done

# logfile
timestamp=$(date +'%Y-%m-%d-%H-%M-%S')
logfile="./logs/st_rl/train_${task_name}_${timestamp}_${device}.log"

echo "time is : $timestamp"
echo "log file is at $logfile"
echo "device is $device"

if $nohup; then
    #python -u ./scripts/st_rl/train.py --task $task_name --headless --device $device --config-name=config1
    #nohup will let the process run in the background
    #nohup python -u ./scripts/st_rl/train.py --task $task_name --headless --device $device --config-name=config1 > $logfile 2>&1 &
    nohup python -u ./scripts/st_rl/train.py --task $task_name --headless $resume --device $device $load_run > $logfile 2>&1 &
else
    python -u ./scripts/st_rl/train.py --task $task_name --headless $resume $load_run --device $device --config-name=config1 > $logfile 2>&1 
fi
