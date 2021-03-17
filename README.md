# Monocular Quasi-Dense 3D Object Tracking

![](imgs/teaser.gif)

Monocular Quasi-Dense 3D Object Tracking (QD-3DT) is an online framework detects and tracks objects in 3D using quasi-dense object proposals from 2D images.



> [**Monocular Quasi-Dense 3D Object Tracking**](https://arxiv.org/abs/2103.07351),            
> Hou-Ning Hu, Yung-Hsu Yang, Tobias Fischer, Trevor Darrell, Fisher Yu, Min Sun,        
> *arXiv technical report ([arXiv 2103.07351](https://arxiv.org/abs/2103.07351))* 
> *Project Website ([QD-3DT](https://eborboihuc.github.io/QD-3DT/))* 


    @article{Hu2021QD3DT,
        author = {Hu, Hou-Ning and Yang, Yung-Hsu and Fischer, Tobias and Yu, Fisher and Darrell, Trevor and Sun, Min},
        title = {Monocular Quasi-Dense 3D Object Tracking},
        journal = {ArXiv:2103.07351},
        year = {2021}
    }

## Abstract

A reliable and accurate 3D tracking framework is essential for predicting future locations of surrounding objects and planning the observer’s actions in numerous applications such as autonomous driving. We propose a framework that can effectively associate moving objects over time and estimate their full 3D bounding box information from a sequence of 2D images captured on a moving platform. The object association leverages quasi-dense similarity learning to identify objects in various poses and viewpoints with appearance cues only. After initial 2D association, we further utilize 3D bounding boxes depth-ordering heuristics for robust instance association and motion-based 3D trajectory prediction for re-identification of occluded vehicles. In the end, an LSTM-based object velocity learning module aggregates the long-term trajectory information for more accurate motion extrapolation. Experiments on our proposed simulation data and real-world benchmarks, including KITTI, nuScenes, and Waymo datasets, show that our tracking framework offers robust object association and tracking on urban-driving scenarios. On the Waymo Open benchmark, we establish the first camera-only baseline in the 3D tracking and 3D detection challenges. Our quasi-dense 3D tracking pipeline achieves impressive improvements on the nuScenes 3D tracking benchmark with near five times tracking accuracy of the best vision-only submission among all published methods.


## Main results

### 3D tracking on nuScenes test set
> We achieved the best vision-only submission

|  AMOTA  |  AMOTP   |
|---------|----------|
|   21.7  |   1.55   |

### 3D tracking on Waymo Open test set
> We established the first camera-only baseline on Waymo Open

| MOTA/L2 | MOTP/L2 |
|---------|---------|
| 0.0001  |  0.0658 |

### 2D vehicle tracking on KITTI test set

|  MOTA   |  MOTP  |
|---------|--------|
| 86.44   |  85.82 |


## Installation

Please refer to [INSTALL.md](./readme/INSTALL.md) for installation and to [DATA.md](./readme/DATA.md) dataset preparation.


## Get Started

Please see [GETTING_STARTED.md](./readme/GETTING_STARTED.md) for the basic usage of QD-3DT.


## MODEL ZOO

Please refer to [MODEL_ZOO.md](./readme/MODEL_ZOO.md) for reproducing the results on varients of benchmarks


## Contact

This repo is currently maintained by Hou-Ning Hu ([@eborboihuc](http://github.com/eborboihuc)), Yung-Hsu Yang ([@RoyYang0714](https://github.com/RoyYang0714)), and Tobias Fischer ([@tobiasfshr](https://github.com/tobiasfshr)).


## License
This work is licensed under BSD 3-Clause License. See [LICENSE](LICENSE) for details. 
Third-party datasets and tools are subject to their respective licenses.

## Acknowledgements
We thank [Jiangmiao Pang](https://github.com/OceanPang) for his help in providing the [qdtrack](https://github.com/SysCV/qdtrack) codebase in [mmdetection](https://github.com/open-mmlab/mmdetection). This repo uses [py-motmetrics](https://github.com/cheind/py-motmetrics) for MOT evaluation, [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) for Waymo Open 3D detection and 3D tracking task, and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for nuScenes evaluation and preprocessing.
