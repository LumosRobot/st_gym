#!/usr/bin/bash
#config: --utf-8

export PYTHONPATH=$(pwd):$(pwd)/third_party:$(pwd)/third_party/refmotion_manager:$PYTHONPATH

load_run=""
task_name="Flat-Lus2"
experiment_name="lus2_flat"
while getopts "n:l:c:e:d:m:h:r" arg; do
  case $arg in
    n)
      task_name=$OPTARG
      ;;
    m)
      run_mode="${OPTARG}"
      ;;
    l)
      load_run="--load_run $OPTARG"
      resume="--resume=True"
      ;;
    h)
      nohuplog="--nohup $OPTARG"
      ;;
    c)
      checkpoint="--checkpoint $OPTARG"
      ;;
    d)
      device="--device $OPTARG"
      ;;
    r)
      export_rknn="--export_rknn"
      ;;
    e)
      experiment_name=$OPTARG
      ;;
    ?)
      echo "Unknown args"
      exit 1
      ;;
  esac
done

    timestamp=$(date +'%Y-%m-%d-%H-%M-%S')
    echo "time is : $timestamp"

if [[ "$run_mode" == *"train"* ]]; then

    logfile="./logs/st_rl/train_${task_name}_${idx}_${timestamp}.log"
    echo "log file is at $logfile"
    nohup python -u ./scripts/st_rl/train.py --task $task_name $load_run $resume $checkpoint $device --headless > "$logfile" 2>&1 &
    tail -f $logfile

elif [[ "$run_mode" == *"play"* ]]; then
    logfile="./logs/st_rl/play_${task_name}_${idx}_${timestamp}.log"
    echo "log file is at $logfile"
  nohup python -u ./scripts/st_rl/play.py --task $task_name  $load_run $checkpoint $otherarg $play_demo --cfg_file load_run --experiment_name $experiment_name > "$logfile" 2>&1 &
    tail -f $logfile

elif [[ "$run_mode" == *"sim"* ]]; then
    echo "mujoco simulation.."
  python ./scripts/st_rl/sim2mujoco.py --task $task_name $load_run --experiment_name $experiment_name $export_rknn
elif [[ "$run_mode" == *"assess"* ]]; then
    echo "eval ..."
  python ./scripts/st_rl/eval_deploy.py --task $task_name $load_run --experiment_name $experiment_name $nohuplog
fi





