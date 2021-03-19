# MODEL ZOO

### Common settings and notes

- The experiments are run with PyTorch 1.4.0, CUDA 10.1, and CUDNN 7.5.
- The models can be downloaded directly from the download links.

## Monocular 3D Detection / Tracking

### nuScenes

| Model                    | GPUs |Train time| Val AMOTA@0.2 | Val AMOTA |  Download | 
|--------------------------|------|----------|---------------|-----------|-----------|
|   nuScenes_3Dtracking    |  8   |   144h   |      0.352    |   0.242   | [model](https://drive.google.com/file/d/14gdw74AcIhvqAEWk3nQwgbGHge_-X1vM/view?usp=sharing)|
|nuScenes_LSTM_motion_model|  1   |    9h    |      -        |  -        | [model](https://drive.google.com/file/d/1tFME2bmPJsNsEtWYP6qS3AW0VZVRCeWo/view?usp=sharing)|

Place the tracking model weight under
```bash
${QD-3DT}/work_dirs/Nusc/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter/latest.pth
```
Place the LSTM model weight under
```bash
${QD-3DT}/checkpoints/batch128_min10_seq10_dim7_VeloLSTM_nuscenes_100_linear.pth
```

#### Notes

- Tracking model is trained on the keyframes of 6 camera images for 24 epochs on servers with 8x 32G V100 GPUs.
- LSTM model is trained on the pure detection results and greedy matching to the groundtruth on servers with 1x 1080Ti GPU.

### Waymo

```diff
! NOTE: Due to Waymo Open Dataset license, we cannot release the models trained on the dataset.
```

| Model                    | GPUs |Train time| Val MOTA/L2 [0m,30m)] |  Download | 
|--------------------------|------|----------|-----------------------|-----------|
|     Waymo_3Dtracking     |  8   |   336h   |         0.0001        | [model]() |
| Waymo_LSTM_motion_model  |  1   |   24h    |           -           | [model]() |

Place the tracking model weight under
```bash
${QD-3DT}/work_dirs/Waymo/quasi_r101_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_scale_no_filter_scaled_res/latest.pth
```
Place the LSTM model weight under
```bash
${QD-3DT}/checkpoints/batch128_min10_seq10_dim7_VeloLSTM_waymo_100_linear.pth
```

#### Notes

- Tracking model is trained on all 5 camera images for 24 epochs on servers with 8x 32G V100 GPUs.
- LSTM model is trained on the pure detection results and greedy matching to the groundtruth on servers with 1x 1080Ti GPU.

### KITTI

| Model                             | GPUs |Train time| MOTA (2D Test) |  Download | 
|-----------------------------------|------|----------|----------------|-----------|
|        KITTI_train_3Dtrack        |  4   |   16h    |      86.41     | [model](https://drive.google.com/file/d/1mwXVr-3B4BxPtxmJF-Sfddq_SjDcT7Kx/view?usp=sharing) |
|     KITTI_LSTM_motion_model       |  1   |   0.5h   |        -       | [model](https://drive.google.com/file/d/10REtqfmkYMCexYaMAy9js-Zp54wOfuO4/view?usp=sharing) |
|      KITTI_subtrain_3Dtrack       |  4   |   14h    |        -       | [model](https://drive.google.com/file/d/1_ikkK3ABE-9fA7Ja7DjQNqLAvBMcaVPi/view?usp=sharing) |
| KITTI_subtrain_LSTM_motion_model  |  1   |   0.5h   |        -       | [model](https://drive.google.com/file/d/1H25HzRcWhtuWk_bOu1_zmzOKjjHaRw0O/view?usp=sharing) |

Place the full train tracking model weight under
```bash
${QD-3DT}/work_dirs/KITTI/quasi_dla34_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_mod_anchor_ratio_small_strides_GTA/latest.pth
```
and half train tracking model weight under
```bash
${QD-3DT}/work_dirs/KITTI/quasi_dla34_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_subtrain_mod_anchor_ratio_small_strides_GTA/latest.pth
```

Place the LSTM model weight under
```bash
${QD-3DT}/checkpoints/batch8_min10_seq10_dim7_train_dla34_regress_pretrain_VeloLSTM_kitti_100_linear.pth
```
and 
```bash
${QD-3DT}/checkpoints/batch8_min10_seq10_dim7_subtrain_dla34_regress_pretrain_VeloLSTM_kitti_100_linear.pth
```

#### Notes

- Tracking model is trained on both detection and tracking images for 24 epochs on servers with 4x 32G V100 GPUs.
- LSTM model is trained on the pure detection results and greedy matching to the groundtruth on servers with 1x 1080Ti GPU
