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
./scripts/test_eval_exp.sh waymo ${config_path} ${gpu_ids} ${gpu_nums} --add_test_set ${PY_ARGS}

# 3D Detection & Tracking Generation
python scripts/waymo_devkit/eval_waymo_3d.py \
--phase=test \
--work_dir=$work_dir \

# create 3D Detection submission
if [ ! -d "${work_dir}/result_3D_DET" ]; then
    mkdir "${work_dir}/result_3D_DET"
fi

./scripts/waymo_devkit/waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/create_submission \
--input_filenames=${work_dir}/result_3D.bin \
--output_filename=${work_dir}/result_3D_DET/submission.bin \
--submission_filename='scripts/waymo_devkit/submission_3d_det.txtpb'

tar zcvf "${work_dir}/result_3D_DET.tar.gz" "${work_dir}/result_3D_DET"

# create 3D Tracking submission
if [ ! -d "${work_dir}/result_3D_MOT" ]; then
    mkdir "${work_dir}/result_3D_MOT"
fi

./scripts/waymo_devkit/waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/create_submission \
--input_filenames=${work_dir}/result_3D.bin \
--output_filename=${work_dir}/result_3D_MOT/submission.bin \
--submission_filename='scripts/waymo_devkit/submission_3d_mot.txtpb'

tar zcvf "${work_dir}/result_3D_MOT.tar.gz" "${work_dir}/result_3D_MOT"
