#!/usr/bin/env bash
set -x
# ----configs----
work_dir=$1
config_path=$2
gpu_ids=$3
gpu_nums=$4
PY_ARGS=${@:5}
# --------------
config=$(basename -s .py ${config_path})
# -----------------

# get the results of each camera
./scripts/test_eval_exp.sh nuscenes ${config_path} ${gpu_ids} ${gpu_nums} ${PY_ARGS}

# 3D Detection Evaluation
python scripts/eval_nusc_det.py \
--version=v1.0-trainval \
--root=data/nuscenes/ \
--work_dir=$work_dir \
--gt_anns=data/nuscenes/anns/tracking_val.json 2>&1 | tee ${work_dir}/eval_det_nusc.txt

# AMOTA@1
export PYENV_VERSION=Nusc

python scripts/eval_nusc_mot.py \
--version=v1.0-trainval \
--root=data/nuscenes/ \
--work_dir=$work_dir \
--gt_anns=data/nuscenes/anns/tracking_val.json 2>&1 | tee ${work_dir}/eval_mot_nusc.txt

# AMOTA@0.2
export PYTHONPATH="${PYTHONPATH}:scripts/nuscenes-devkit/python-sdk"

python scripts/eval_nusc_mot.py \
--version=v1.0-trainval \
--root=data/nuscenes/ \
--work_dir=$work_dir \
--gt_anns=data/nuscenes/anns/tracking_val.json \
--amota_02 2>&1 | tee ${work_dir}/eval_mot_02_nusc.txt
