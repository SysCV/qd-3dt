#!/usr/bin/env bash
set -x
# ----configs----
config_path=$1
gpu_ids=$2
gpu_nums=$3
resume=${4:-0}
# --------------
root=.
folder='work_dirs/'$(dirname ${config_path#*/})
config=$(basename -s .py ${config_path})
# -----------------

if [[ $resume = *"resume"* ]]; then
    resume_from=./${folder}/${config}/latest.pth
else
    resume_from=""
fi

cd mmcv
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..

# train
CUDA_VISIBLE_DEVICES=${gpu_ids} python3 -u ${root}/tools/train.py \
${config_path} \
--work_dir=./${folder}/${config} \
--resume_from=${resume_from} \
--gpus=${gpu_nums} \
--validate \

echo "Config: " ${config}
cp ${config_path} ./${folder}/${config}/.
