# GenSafeNav-ROS2

This is the ROS2 deployment codebase for the paper: _[Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling](https://arxiv.org/abs/2508.05634v1)_.

For more information, please also check:

1.) [Project website](https://gen-safe-nav.github.io/)

2.) [Video demos](https://youtu.be/z8Eux3UOWc8)

3.) [Training code](https://github.com/tasl-lab/GenSafeNav)

## Overview

This repository contains the ROS2 system for deploying the GenSafeNav policy on a real robot. It integrates pedestrian detection, tracking, trajectory prediction, and RL-based decision making for safe crowd navigation.

## Components

```
.
├── decider/                 # RL-based decision making module
│   ├── decider/             # Main ROS2 node
│   ├── rl/networks/         # Policy network (selfAttn_srnn)
│   ├── config/              # Configuration files
│   └── model_weight/        # Pre-trained model (ours.pt)
├── predictor/               # Trajectory prediction module
│   ├── predictor/           # Main ROS2 node with DtACI
│   └── gst_updated/         # Gumbel Social Transformer
├── dr_spaam_ros2/           # 2D LiDAR person detection (DR-SPAAM)
├── sort_tracker/            # Multi-object tracking (SORT)
├── command_listener/        # User command interface
├── frequency_monitor/       # System performance monitoring
└── fake_detection/          # Simulation utilities
```

## Setup

This package requires ROS2 (tested on **Foxy**). Clone into your ROS2 workspace and build using the standard procedure:

```bash
cd ~/ros2_ws/src
git clone <repository_url>
cd ..
colcon build
source install/setup.bash
```

Before running, update the configuration files (e.g., `dr_spaam_ros2/config/dr_spaam_ros2.yaml`) with your local paths.

**Note:** For best results when reproducing real-world experiments, we recommend testing in a large open space.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{yao2025towards,
    title={Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling},
    author={Yao, Jianpeng and Zhang, Xiaopan and Xia, Yu and Roy-Chowdhury, Amit K and Li, Jiachen},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2025}
}
```

## Acknowledgement

We sincerely thank the researchers and developers for [CrowdNav](https://github.com/vita-epfl/CrowdNav), [CrowdNav++](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph), [Gumbel Social Transformer](https://sites.google.com/view/gumbel-social-transformer), [DtACI](https://github.com/isgibbs/DtACI), [DR-SPAAM](https://github.com/VisualComputingInstitute/DR-SPAAM-Detector), and [OmniSafe](https://github.com/PKU-Alignment/omnisafe) for their amazing work.
