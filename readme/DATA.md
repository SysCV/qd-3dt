# Dataset preparation
If you want to reproduce the results in the paper for benchmark evaluation or training, you will need to setup datasets.

## nuScenes
nuScenes is used for training and evaluating 3D object tracking.

- Download the dataset from [nuScenes website](https://www.nuscenes.org/download?externalData=all&mapData=all&modalities=Any). You also need to download the maps and all metadata to make the nuScenes API work.

- Unzip, and place (or symlink) the data as below. You will need to merge folders from different zip files.

```bash
${QD-3DT_ROOT}
|-- data
`-- |-- nuscenes
    `-- |-- maps
        |-- samples
        |   |-- CAM_BACK
        |   |   | -- xxx.jpg
        |   |-- CAM_BACK_LEFT
        |   |-- CAM_BACK_RIGHT
        |   |-- CAM_FRONT
        |   |-- CAM_FRONT_LEFT
        |   `-- CAM_FRONT_RIGHT
        |-- sweeps
        |   |-- CAM_BACK
        |   |-- CAM_BACK_LEFT
        |   |-- CAM_BACK_RIGHT
        |   |-- CAM_FRONT
        |   |-- CAM_FRONT_LEFT
        |   `-- CAM_FRONT_RIGHT
        |-- v1.0-mini
        |-- v1.0-trainval
        `-- v1.0-test
```

- Run ``convert_nuScenes.py`` in ``scripts`` to convert the annotation into COCO format. It will create ``tracking_train_mini.json``, ``tracking_val_mini.json``, ``tracking_train.json``, ``tracking_val.json``, ``tracking_test.json`` under ``data/nuscenes/anns``. nuScenes API is required for running the data preprocessing.

```bash
$ python scripts/convert_nuScenes.py
```

- Run ``convert_nuScenes_full_frames.py`` in ``scripts`` to convert full frames information into COCO format for inference. It will create ``tracking_val_mini_full_frames.json``, ``tracking_val_full_frames.json``, ``tracking_test_full_frames.json`` under ``data/nuscenes/anns``. nuScenes API is also required for running the data preprocessing.

```bash
$ python scripts/convert_nuScenes_full_frames.py
```

## Waymo Open
Waymo is used for training and evaluating 3D object tracking.

- Download the dataset from [Waymo Open Datset website](https://waymo.com/open/download/). You can download individual files or tar files (training, validation, testing).

- Unzip (for tar files), and place (or symlink) the data as below. Note that the gt.bin is for validation evaluation and under ``validation/groud_truth_objects`` folder.

```bash
${QD-3DT_ROOT}
|-- data
`-- |-- Waymo
    `-- |-- raw
        |   |-- training
        |   |   | -- xxx.tfrecord
        |   |-- validation
        |   `-- testing
        `-- gt.bin
```

- Run ``convert_Waymo.py`` in ``scripts`` to convert the annotation into COCO format. It will create ``tracking_train_mini.json``, ``tracking_val_mini.json``, ``tracking_train.json``, ``tracking_val.json``, ``tracking_test.json`` under ``data/Waymo/anns``. It will also create folder to save Waymo images under ``data/Waymo/images_png`` so be sure the storage is big enough for images. ``waymo-open-dataset`` is required for running the data preprocessing.

```bash
$ python scripts/convert_Waymo.py
```

- Run ``generate_waymo_gt.py`` in ``scripts/waymo_devkit`` to generate different ground truth for evaluation. It will create ``gt_mini.bin``, ``gt_mini_projected.bin``, ``gt_validation.bin`` (same as provided ``gt.bin``), ``gt_validation_projected.bin`` under ``data/Waymo``.

```bash
$ python scripts/waymo_devkit/generate_waymo_gt.py
```

##  KITTI Tracking

- Download images, annotations, oxts and calibration information from [KITTI Tracking website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and [KITTI Detection website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 

- Unzip (for zip files), and place (or symlink) the data as below.
  
```bash
${QD-3DT_ROOT}
|-- data
    |-- KITTI
        |-- tracking
        |   |-- training
        |   |   |-- image_02
        |   |   |   |-- 0000
        |   |   |   |   |-- 000000.png
        |   |   |   |   `-- ...
        |   |   |   `-- ...
        |   |   |-- label_02
        |   |   |   |-- 0000.txt
        |   |   |   `-- ...
        |   |   |-- calib
        |   |   |   |-- 0000.txt
        |   |   |   `-- ...
        |   |   `-- oxts
        |   |       |-- 0000.txt
        |   |       `-- ...
        |   `-- testing
        |       |-- image_02
        |       |   |-- 0000
        |       |   |   |-- 000000.png
        |       |   |   `-- ...
        |       |   `-- ...
        |       |-- calib
        |       |   |-- 0000.txt
        |       |   `-- ...
        |       `-- oxts
        |           |-- 0000.txt
        |           `-- ...
        `-- detection
            |-- training
            |   |-- image_2
            |   |   |-- 000000.png
            |   |   `-- ...
            |   |-- label_2
            |   |   |-- 000000.txt
            |   |   `-- ...
            |   |-- calib
            |   |   |-- 000000.txt
            |   |   `-- ...   
            `-- testing
                |-- image_2
                |   |-- 000000.png
                |   `-- ...
                `-- calib
                    |-- 000000.txt
                    `-- ... 
```

- Run ``kitti2coco.py`` in ``scripts`` to convert the annotation into COCO format. 
- It will create ``tracking_subval_mini.json``, ``tracking_subval.json``, ``tracking_train.json``, ``tracking_subtrain.json``, ``tracking_test.json`` and similar files with prefix ``detection-`` under ``data/KITTI/anns``.

```bash
$ python scripts/kitti2coco.py
```

- Copy or Soft link the `*.seqmap` under `scripts/object_ap_eval/seqmaps/` to `data/KITTI/anns/` for later evaluation

```bash
# Under ${QD-3DT_ROOT}
ln -sr scripts/object_ap_eval/seqmaps/*.seqmap data/KITTI/anns
```


- The resulting data structure should look like:

```bash
${QD-3DT_ROOT}
|-- data
    |-- KITTI
        |-- tracking
        |-- detection
        `-- anns
            |-- tracking_train.json
            |-- tracking_train.json.seqmap
            |-- tracking_test.json
            `-- ...
```

# References
```
    @inproceedings{nuscenes2019,
        title = {{nuScenes}: A multimodal dataset for autonomous driving},
        author = {Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
        booktitle = {CVPR},
        year = {2020}
    }
    @inproceedings{sun2020scalability,
        title = {Scalability in perception for autonomous driving: Waymo open dataset},
        author = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages = {2446--2454},
        year = {2020}
    }
    @INPROCEEDINGS{Geiger2012CVPR,
        author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
        title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
        booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2012}
    }
```
