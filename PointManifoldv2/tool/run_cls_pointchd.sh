#!/bin/sh
PARTITION=gpu
PYTHON=python

num_points=$1
model_name=$2
epochs=$3
lr=$4
exp_name=cls_${num_points}_${model_name}_lr${lr}_e${epochs}
now=$(date +"%Y%m%d_%H%M%S")

exp_dir=outputs/${exp_name}
model_dir=${exp_dir}/models

mkdir -p ${exp_dir} ${model_dir}

cp tool/main_cls_pointchd.py model/main_models.py ${exp_dir}

export PYTHONPATH=./
$PYTHON -u tool/main_cls_pointchd.py \
  --exp_name=${exp_name} \
  --num_points=${num_points} \
  --model=${model_name} \
  --epochs=${epochs} \
  --lr=${lr} \
  2>&1 | tee ${exp_dir}/main_cls_pointchd-train-$now.log

$PYTHON -u tool/main_cls_pointchd.py \
  --exp_name=${exp_name} \
  --num_points=${num_points} \
  --model=${model_name} \
  --eval=True \
  --model_path=${exp_dir}/models/model.t7 \
 --seg_result='/home/dinghow/Documents/PointCHD_benchmark/outputs/partseg_2048_pointmanifold2_lr1e-3_e200/result/partseg_2048_pointmanifold2_lr1e-3_e200.json' \
  2>&1 | tee ${exp_dir}/main_cls_pointchd-eval-$now.log
