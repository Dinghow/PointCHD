#!/bin/sh
PARTITION=gpu
PYTHON=python

num_points=$1
model_name=$2
epochs=$3
lr=$4
exp_name=partseg_${num_points}_${model_name}_lr${lr}_e${epochs}
now=$(date +"%Y%m%d_%H%M%S")

exp_dir=outputs/${exp_name}
model_dir=${exp_dir}/models

mkdir -p ${exp_dir} ${model_dir}

cp tool/main_partseg_pointchd.py model/main_models.py ${exp_dir}

export PYTHONPATH=./
$PYTHON -u tool/main_partseg_pointchd.py \
  --exp_name=${exp_name} \
  --model=${model_name} \
  --num_points=${num_points} \
  --epochs=${epochs} \
  --lr=${lr} \
  --scheduler=step \
  2>&1 | tee ${exp_dir}/main_partseg_pointchd-train-$now.log

$PYTHON -u tool/main_partseg_pointchd.py \
  --exp_name=${exp_name} \
  --model=${model_name} \
  --num_points=${num_points} \
  --eval=True \
  --model_path=${exp_dir}/models/model.t7 \
  2>&1 | tee ${exp_dir}/main_partseg_pointchd-eval-$now.log
