# Getting Started
This document provides tutorials to train and evaluate Quasi-Dense 3D Detection and Tracking. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

# Benchmark evaluation
First, download the models you want to evaluate from our [model zoo](MODEL_ZOO.md) and put them in the corresponding directory following the description.

## nuScenes
- Inference on Validation set with full frames information. It will show your 3D detection and 3D Tracking score (both AMOTA@1 and AMOTA@0.2) on validation set.
```
$ ./scripts/run_eval_nusc.sh ${WORK_DIR} ${CONFIG} ${gpu_id} 1 --data_split_prefix ${EXP_NAME} --full_frames

# ${WORK_DIR} is the path to place the model output.
# ${CONFIG} is the corresponding config file you use.
# ${EXP_NAME} is the experiment name you want to specify.
```

- To reproduce our result, please run the following code:
```
$ ./scripts/run_eval_nusc.sh work_dirs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter/output_val_box3d_deep_depth_motion_lstm_3dcen configs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py 0 1 --data_split_prefix val --full_frames
```

- Inference on Test set with full frames information. It will generate the result json files (``detection_result.json`` & ``tracking_result.json``) under ``${WORK_DIR}`` to submit to the benchmark server.
```
$ ./scripts/run_test_nusc.sh ${WORK_DIR} ${CONFIG} ${gpu_id} 1 --data_split_prefix ${EXP_NAME}
```

- To reproduce our result, please run the following code:
```
$ ./scripts/run_test_nusc.sh work_dirs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter/output_test_box3d_deep_depth_motion_lstm_3dcen configs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py 0 1 --data_split_prefix test --full_frames
```

## Waymo
- Inference on Validation set. It will show your 3D detection and 3D Tracking score on validation set and generate the result tar files (``result_3D_DET.tar.gz`` and ``result_3D_MOT.tar.gz``) under ``${WORK_DIR}`` to submit to the validation server.

```
$ ./scripts/run_eval_waymo.sh ${WORK_DIR} ${CONFIG} ${gpu_id} 1 --data_split_prefix ${EXP_NAME}

# ${WORK_DIR} is the path to place the model output.
# ${CONFIG} is the corresponding config file you use.
# ${EXP_NAME} is the experiment name you want to specify.
```

- To reproduce our result, please run the following code:
```
$ ./scripts/run_eval_waymo.sh work_dirs/Waymo/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter_scaled_res/output_val_box3d_deep_depth_motion_lstm_3dcen configs/Waymo/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter_scaled_res.py 0 1 --data_split_prefix val
```

- Inference on Test set. It will generate the result tar files (``result_3D_DET.tar.gz`` and ``result_3D_MOT.tar.gz``) under ``${WORK_DIR}`` to submit to the benchmark server.
```
$ ./scripts/run_test_waymo.sh ${WORK_DIR} ${CONFIG} ${gpu_id} 1 --data_split_prefix ${EXP_NAME}

# ${WORK_DIR} is the path to place the model output.
# ${CONFIG} is the corresponding config file you use.
# ${EXP_NAME} is the experiment name you want to specify.
```

- To reproduce our result, please run the following code:
```
$ ./scripts/run_test_waymo.sh work_dirs/Waymo/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter_scaled_res/output_test_box3d_deep_depth_motion_lstm_3dcen configs/Waymo/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter_scaled_res.py 0 1 --data_split_prefix test
```

## KITTI
- Inference on sub-val set. It will show your 3D detection and 3D Tracking score on validation set and generate the result txt files (``txts/{:04}.txt``) under ``${WORK_DIR}`` for further offline evaluation.

```bash
$ ./scripts/test_eval_exp.sh ${dataset} ${CONFIG} ${gpu_id} 1 --data_split_prefix ${EXP_NAME} --add_ablation_exp ${FLAG}

# ${WORK_DIR} is the path to place the model output.
# ${CONFIG} is the corresponding config file you use.
# ${EXP_NAME} is the experiment name you want to specify.
# ${FLAG} is the experiment flag from tools/test_eval_video_exp.py
```

- To reproduce our result, please run the following code:
```bash
$ ./scripts/test_eval_exp.sh kitti configs/KITTI/quasi_dla34_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_subtrain_mod_anchor_ratio_small_strides_GTA.py 0 1 --data_split_prefix subval_dla34_regress_GTA_VeloLSTM --add_ablation_exp all
```

```bash
$ ./scripts/test_eval_exp.sh ${dataset} ${CONFIG} ${gpu_id} 1 --data_split_prefix ${EXP_NAME} --add_ablation_exp ${FLAG}

# ${WORK_DIR} is the path to place the model output.
# ${CONFIG} is the corresponding config file you use.
# ${EXP_NAME} is the experiment name you want to specify.
# ${FLAG} is the experiment flag from tools/test_eval_video_exp.py
```

- To reproduce our result, please run the following code:
```bash
$ ./scripts/test_eval_exp.sh kitti configs/KITTI/quasi_dla34_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_mod_anchor_ratio_small_strides_GTA.py 0 1 --data_split_prefix test_dla34_regress_GTA_VeloLSTM --add_test_set
```

# Training
## Tracking model
- Train on the dataset you want by specifying the corresponding config file.
```bash
./scripts/train.sh ${CONFIG} ${gpu_id} ${gpu_nums}
```

- If the training is terminated before finish, you can add ``--resume`` to the command above to resume the training.
```bash
./scripts/train.sh ${CONFIG} ${gpu_id} ${gpu_nums} --resume
```

## LSTM motion model training
- Get the pure detection result from the required dataset for both training and validation set after tracking model finish training. It will generate results as ``work_dirs/${DATASET}/{CONFIG}/output_train_pure_det_3dcen/output.json`` or ``work_dirs/${DATASET}/{CONFIG}/output_val_pure_det_3dcen/output.json``.
```bash
./scripts/test_eval_exp.sh ${dataset} ${CONFIG} ${gpu_id} 1 --data_split_prefix ${EXP_NAME} --pure_det

# ${EXP_NAME} stands for which set is to be generated for LSTM motion model and only accept train / val.
```
### Take nuScenes as example
- To reproduce our result, please run the following code:
```bash
# get the pure detection result for training set
./scripts/test_eval_exp.sh nuscenes configs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py 0 1 --data_split_prefix train --pure_det

# get the pure detection result for validation set
./scripts/test_eval_exp.sh nuscenes configs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter.py 0 1 --data_split_prefix val --pure_det
```

- Soft-link pure detection results under ``data/${DATASET}/anns`` as ``tracking_output_train.json`` and ``tracking_output_val.json``.
```bash
# nuScenes example
${QD-3DT_ROOT}
|-- data
`-- |-- nuscenes
    `-- |-- anns
        `-- |-- tracking_output_train.json
            |-- tracking_output_val.json
            |-- tracking_train.json
            |-- tracking_val.json
```

- Train LSTM motion model for required dataset.
```bash
CUDA_VISIBLE_DEVICES=${gpu_id} python qd3dt/models/detectrackers/tracker/motion_lstm.py ${DATASET} train \
--session batch128_min10_seq10_dim7_VeloLSTM \
--min_seq_len 10 --seq_len 10 \
--lstm_model_name VeloLSTM --tracker_model_name KalmanBox3DTracker \
--input_gt_path ${GT_training} \
--input_pd_path ${pure_det_training} \
--cache_name work_dirs/LSTM/nuscenes_train_pure_det_min10.pkl \
--loc_dim 7 -b 128 --is_plot --show_freq 500

# ${GT_training} is the corresponding training annotation file.
# ${pure_det_training} is the corresponding pure detection result for training set.
```

- To reproduce our result, please run the following code:
```bash
CUDA_VISIBLE_DEVICES=1 python qd3dt/models/detectrackers/tracker/motion_lstm.py nuscenes train \
--session batch128_min10_seq10_dim7_VeloLSTM \
--min_seq_len 10 --seq_len 10 \
--lstm_model_name VeloLSTM --tracker_model_name KalmanBox3DTracker \
--input_gt_path data/nuscenes/anns/tracking_train.json \
--input_pd_path data/nuscenes/anns/tracking_output_train.json \
--cache_name work_dirs/LSTM/nuscenes_train_pure_det_min10.pkl \
--loc_dim 7 -b 128 --is_plot --show_freq 500
```
