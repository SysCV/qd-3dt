# Installation Guide

## Prerequisites
The main branch works with **PyTorch 1.3.1** or higher.

- Linux (tested on Ubuntu 16.04.4 LTS)
- Python: 3.7.6
    - `3.6.9` tested
    - `3.7.6` tested
- PyTorch: 1.4.0
    - `1.3.1` (with CUDA 10.1, torchvision 0.4.2)
    - `1.4.0` (with CUDA 10.1, torchvision 0.5.0)
- nvcc: 10.1
    - `9.1.85`, `10.1.243` compiling and execution tested
- gcc: 5.4.0
    - `5.4.0`
    - `5.5.0`
- Nvidia driver version: 440.59
- Cudnn version: 7.6.4
- Pyenv or Anaconda

Python dependencies list in `requirements.txt` 


## Installation
- Clone this repo:
```bash
git clone -b main --single-branch https://github.com/SysCV/qd-3dt.git
```

- Create folders
```bash
cd qd-3dt/
mkdir data work_dirs checkpoints
```

- You can create a virtual environment by the following:

We use [pyenv](https://github.com/pyenv/pyenv#installation) as our Python version management tool. If you have experienced some installation issue, please refer to [Troubleshooting](https://github.com/pyenv/pyenv/wiki#suggested-build-environment). You may also use [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for the following installation.

```bash
# Add path to bashrc 
echo -e '\nexport PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

# Install pyenv
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# Restart a new terminal if "exec $SHELL" doesn't work
exec $SHELL

# Install and activate Python in pyenv
pyenv install 3.7.6
```

- Create virtualenv for training, inference and evaluation.
```bash
pyenv virtualenv 3.7.6 qd_3dt
pyenv local qd_3dt
```

- Install PyTorch 1.4.0 and torchvision from http://pytorch.org and other dependencies. 
```bash
pyenv activate qd_3dt
pip install -r requirements.txt
```

- Install requirements, create folders and compile binaries for detection under ``qd_3dt`` virtualenv.
```bash
bash install.sh
```

## nuScenes
### Build new virtualenv for nuScenes 3D tracking evaluation.

- Create virtualenv and install nuScenes toolkit with motmetrics==1.13.0 for AMOTA evaluation.
```bash
pyenv virtualenv 3.7.6 Nusc
pyenv activate Nusc
pip install nuscenes-devkit
pip install -r requirements_nusc.txt
pyenv deactivate
```

- Install AMOTA@0.2 evaluation python-sdk.
```bash
cd scripts/
git clone https://github.com/nutonomy/nuscenes-devkit
cd nuscenes-devkit

# Check to the commit for AMOTA@0.2 evaluation
git checkout e2d8c4b331567dc0bc36271dc21cdef65970eb7e
cd ../../
```

## Waymo
### Build the [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) for local evaluation.

- First follow [local compilation](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md#local-compilation-without-docker-system-requirements) to setup waymo-open-dataset toolkit under `scripts/waymo_devkit/`.

- NOTE: you need to checkout to commit `b2b2fc8f06ed6801aec1ea2d406d559bff08b6b5` instead of `remotes/origin/master` for successful installation.

- Compile required evaluation metric functions for both detection and tracking.
```bash
cd scripts/waymo_devkit/waymo-od

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
bazel build waymo_open_dataset/metrics/tools/compute_tracking_metrics_main
bazel build waymo_open_dataset/metrics/tools/create_submission
```

- Create and place the [submission file](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/metrics/tools/submission.txtpb) for your task.
```bash
${QD-3DT_ROOT}
|-- scripts
`-- |-- waymo_devkit
    `-- |-- waymo-od
        |-- eval_waymo_3d.py
        |-- generate_waymo_gt.py
        |-- submission_3d_det.txtpb
        `-- submission_3d_mot.txtpb
```
