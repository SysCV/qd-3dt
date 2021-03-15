#!/usr/bin/env bash
set -x
# DATASET: dataset want to visulization, choice: kitti, nuscenes, gta, waymo
# LABEL: path to label files, data/KITTI/anns/tracking_all-C_subval.json 
# FOLDER: result folder e.g., work_dirs/KITTI/quasi_r50_dcn_3dmatch_multibranch_conv_dep_dim_rot2/output/txts/
# --res_folder ${FOLDER} --is_gt --align_gt

DATASET=$1
LABEL=$2
PY_ARGS=${@:3}

cd mmcv
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..

python scripts/plot_tracking.py \
${DATASET} \
${LABEL} \
--draw_3d \
--draw_bev \
--draw_traj \
--is_merge \
--is_save \
${PY_ARGS}
