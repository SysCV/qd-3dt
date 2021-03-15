#!/usr/bin/env bash
set -x
# ----configs----
work_dir=$1
config_path=$2
gpu_ids=$3
gpu_nums=$4
PY_ARGS=${@:5}
# --------------
folder='work_dirs/'$(dirname ${config_path#*/})
config=$(basename -s .py ${config_path})
# -----------------

# get the results of each camera
./scripts/test_eval_exp.sh nuscenes ${config_path} ${gpu_ids} ${gpu_nums} --add_test_set ${PY_ARGS}

# 3D Detection generation
python scripts/eval_nusc_det.py \
--version=v1.0-test \
--root=data/nuscenes/ \
--work_dir=$work_dir \
--gt_anns=data/nuscenes/anns/tracking_test.json

# 3D Tracking generation
python scripts/eval_nusc_mot.py \
--version=v1.0-test \
--root=data/nuscenes/ \
--work_dir=$work_dir \
--gt_anns=data/nuscenes/anns/tracking_test.json
