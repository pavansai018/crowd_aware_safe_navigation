# GenSafeNav — Crowd-Aware Safe Navigation

> **Paper:** [Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling](https://arxiv.org/abs/2508.05634v1) — CoRL 2025  
> **Demo:** [Video](https://youtu.be/z8Eux3UOWc8) · **Website:** [gen-safe-nav.github.io](https://gen-safe-nav.github.io/) · **Training code:** [GenSafeNav](https://github.com/tasl-lab/GenSafeNav)

GenSafeNav navigates a mobile robot safely through pedestrian crowds. It uses a 2D LiDAR to detect people, SORT to track them across frames, a Gumbel Social Transformer with Adaptive Conformal Inference to predict their trajectories with calibrated uncertainty, and a constrained RL policy to choose velocity commands that reach the goal while respecting safety margins.

---

## Pipeline

```
2D LiDAR (/scan)
     │
     ▼  LaserScan
┌─────────────────┐
│  dr_spaam_ros2  │  detects pedestrians
└────────┬────────┘
         │  PoseArray (/dr_spaam_detections)
         ▼
┌─────────────────┐
│  sort_tracker   │  assigns persistent IDs
└────────┬────────┘
         │  JSON (/tracked_objects_json)
         ├─────────────────────────────┐
         ▼                             ▼
┌─────────────────┐           ┌──────────────┐
│   predictor     │           │   decider    │
│  (GST + DtACI)  │──────────▶│  (RL policy) │──▶ /cmd_vel ──▶ Robot base
└─────────────────┘           └──────────────┘
   /predictions_json           ▲
                               │  /goal_pose, /odom,
                               │  /joint_states, /command
```

---

## Hardware Requirements

### Robot base
- Differential-drive or omnidirectional mobile robot
- Publishes: `/odom` (`nav_msgs/Odometry`), `/joint_states` (`sensor_msgs/JointState`)
- Accepts: `/cmd_vel` (`geometry_msgs/Twist`)
- Required TF frames: `base_link` → `odom`

### 2D LiDAR
- 360° panoramic scan strongly recommended (e.g., RPLiDAR A3, Hokuyo UTM-30LX-EW)
- Publishes: `/scan` (`sensor_msgs/LaserScan`)
- Set `panoramic_scan: false` in config if your sensor covers less than 360°

### Compute
- NVIDIA GPU with CUDA 12.1 support (tested on RTX 2080, Jetson AGX Xavier)
- Minimum 8 GB RAM; 16 GB recommended

---

## Software Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 20.04 |
| ROS2 | Foxy |
| Python | 3.8+ |
| PyTorch | 2.3.1 (CUDA 12.1) |
| colcon | latest |

---

## Repository Layout

```
crowd_aware_safe_navigation/
├── 2D_lidar_person_detection/
│   └── dr_spaam/                  ← install this as a Python package (step 1)
├── GenSafeNav/                    ← training only, not needed for deployment
└── m3_ws/                         ← your ROS2 workspace
    └── src/GenSafeNav-ROS2/
        ├── dr_spaam_ros2/         ← detection node + config
        ├── sort_tracker/          ← tracking node + config
        ├── predictor/             ← prediction node (GST + DtACI)
        ├── decider/               ← RL navigation node
        ├── command_listener/      ← operator keyboard interface
        └── frequency_monitor/     ← optional latency monitor
```

**All model weights are bundled in the repository.** No separate download is needed.

| Package | Weight file |
|---------|-------------|
| DR-SPAAM detector | `2D_lidar_person_detection/dr_spaam/dr_spaam/ckpt_jrdb_ann_ft_dr_spaam_e20.pth` |
| Trajectory predictor | `m3_ws/src/GenSafeNav-ROS2/predictor/model_weight/checkpoint/epoch_100.pt` |
| Navigation policy | `m3_ws/src/GenSafeNav-ROS2/decider/model_weight/ours.pt` |

---

## Installation

### 1. Install the DR-SPAAM Python library

The detection node depends on the `dr_spaam` package, which must be installed from source before building the ROS2 workspace.

```bash
cd 2D_lidar_person_detection/dr_spaam
pip install -e .
```

### 2. Install Python dependencies

```bash
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy filterpy
```

### 3. Build the ROS2 workspace

```bash
cd m3_ws
source /opt/ros/foxy/setup.bash
colcon build
source install/setup.bash
```

---

## Configuration

Only one file must be edited before the system will run.

### Required — DR-SPAAM model path

Open `m3_ws/src/GenSafeNav-ROS2/dr_spaam_ros2/config/dr_spaam_ros2.yaml` and replace the `weight_file` value with the **absolute path** to the checkpoint on your machine:

```yaml
dr_spaam_ros2:
  ros__parameters:
    weight_file: "/absolute/path/to/crowd_aware_safe_navigation/2D_lidar_person_detection/dr_spaam/dr_spaam/ckpt_jrdb_ann_ft_dr_spaam_e20.pth"
    detector_model: "DR-SPAAM"  # or "DROW3"
    conf_thresh: 0.5            # detection confidence threshold (0–1)
    stride: 5                   # skip N laser points per step; higher = faster, less accurate
    panoramic_scan: true        # set false for non-360° LiDAR
```

After editing, rebuild the affected package:

```bash
cd m3_ws
colcon build --packages-select dr_spaam_ros2
source install/setup.bash
```

### Optional — LiDAR topic name

If your LiDAR does not publish on `/scan`, edit `m3_ws/src/GenSafeNav-ROS2/dr_spaam_ros2/config/topics.yaml`:

```yaml
subscriber:
  scan:
    topic: /scan   # ← change to your LiDAR topic
```

Rebuild `dr_spaam_ros2` again after any config change.

---

## Running the System

Source the workspace in every terminal before running:

```bash
source /opt/ros/foxy/setup.bash
source m3_ws/install/setup.bash
```

Start the nodes in this order, one per terminal:

**Terminal 1 — Person detection**
```bash
ros2 launch dr_spaam_ros2 dr_spaam_ros2.launch.py
```

**Terminal 2 — Object tracking**
```bash
ros2 launch sort_tracker sort_tracker.launch.py
```

**Terminal 3 — Trajectory prediction**
```bash
ros2 launch predictor predictor.launch.py
```

**Terminal 4 — Navigation policy**
```bash
ros2 launch decider decider.launch.py
```

**Terminal 5 — Operator interface**
```bash
ros2 launch command_listener command_listener.launch.py
```

**Terminal 6 — Pipeline monitor (optional but recommended)**
```bash
ros2 launch frequency_monitor frequency_monitor.launch.py
```

---

## Operating the Robot

### Step 1 — Select a control mode

When `command_listener` starts, it prompts you to choose:

```
Which mode do you want to choose? (1) Manual (2) Automatic (3) Combined mode
Enter 1, 2, or 3:
```

| Mode | Behavior |
|------|----------|
| `1` Manual | WASD keyboard only; the RL policy is inactive |
| `2` Automatic | RL policy navigates autonomously; keyboard ignored |
| `3` Combined | RL policy navigates; a WASD key press overrides it for 200 ms, then autonomy resumes |

**Combined mode is recommended for real deployments.** It allows the operator to intervene at any moment while the policy handles normal navigation.

### Step 2 — Set a navigation goal

Publish a target pose to `/goal_pose`. From another terminal:

```bash
ros2 topic pub /goal_pose geometry_msgs/PoseStamped "{
  header: {frame_id: 'map'},
  pose: {
    position: {x: 5.0, y: 0.0, z: 0.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}" --once
```

Or use RViz: add a **PoseStamped** display and publish a goal with the **2D Nav Goal** tool — remap the output topic to `/goal_pose`.

---

## Topic Reference

| Topic | Message type | Direction | Approx. Hz | Purpose |
|-------|-------------|-----------|-----------|---------|
| `/scan` | `LaserScan` | LiDAR → dr_spaam | ~30 | Raw sensor data |
| `/dr_spaam_detections` | `PoseArray` | dr_spaam → sort_tracker | ~30 | Detected pedestrian positions |
| `/dr_spaam_rviz` | `MarkerArray` | dr_spaam → RViz | ~30 | Detection visualization |
| `/tracked_objects_json` | `String` | sort_tracker → predictor, decider | ~10 | Tracked IDs + XY positions (JSON) |
| `/tracked_objects_viz` | `MarkerArray` | sort_tracker → RViz | ~10 | Track visualization |
| `/predictions_json` | `String` | predictor → decider | ~10 | Predicted trajectories + uncertainty (JSON) |
| `/predicted_trajectories_viz` | `MarkerArray` | predictor → RViz | ~10 | Prediction visualization |
| `/predicted_trajectories_aci_viz` | `MarkerArray` | predictor → RViz | ~10 | Uncertainty envelope visualization |
| `/odom` | `Odometry` | Robot → decider | ~20 | Robot velocity |
| `/joint_states` | `JointState` | Robot → decider | ~10 | Robot kinematics |
| `/goal_pose` | `PoseStamped` | Operator → decider | on set | Navigation target |
| `/command` | `String` | command_listener → decider | manual | Mode selection and WASD input |
| `/cmd_vel` | `Twist` | decider → Robot | ~10 | Velocity commands |
| `/decider_action_marker` | `Marker` | decider → RViz | ~10 | Current action vector |
| `/decider_goal_marker` | `Marker` | decider → RViz | ~10 | Goal position |
| `/decider_robot_marker` | `Marker` | decider → RViz | ~10 | Robot position |

---

## Tuning

### SORT tracker (`sort_tracker/config/sort_tracker.yaml`)

```yaml
sort:
  max_age: 8          # frames to keep a track alive without a new detection
  min_hits: 4         # detections required before a track is confirmed
  iou_threshold: 0.01 # association threshold; very low suits sparse 2D LiDAR
```

- In dense crowds, increase `max_age` to survive detection gaps.
- Decrease `min_hits` to `1`–`2` for faster confirmation in sparse environments.

### DR-SPAAM detector (`dr_spaam_ros2/config/dr_spaam_ros2.yaml`)

```yaml
conf_thresh: 0.5   # lower → more detections but more false positives
stride: 5          # increase on Jetson to reduce compute; decrease for better accuracy
```

---

## Visualization in RViz

Add these displays to see the full pipeline:

| Display | Topic |
|---------|-------|
| Raw detections | `/dr_spaam_rviz` |
| Tracked objects | `/tracked_objects_viz` |
| Predicted paths | `/predicted_trajectories_viz` |
| Uncertainty envelopes | `/predicted_trajectories_aci_viz` |
| Robot action vector | `/decider_action_marker` |
| Navigation goal | `/decider_goal_marker` |
| Robot marker | `/decider_robot_marker` |

---

## Pipeline Latency

Measured on an RTX 2080:

| Stage | Latency |
|-------|---------|
| LiDAR → Detection | ~30 ms |
| Detection → Tracking | ~10 ms |
| Tracking → Prediction | ~100 ms |
| Prediction → Command | ~100 ms |
| **End-to-end** | **~240 ms** |

The `frequency_monitor` node writes per-topic Hz and end-to-end delay to CSV files in `ros2_frequency_log_[timestamp]/` for offline analysis.

---

## Troubleshooting

**DR-SPAAM node crashes on startup**  
The `weight_file` path in `dr_spaam_ros2.yaml` is wrong or relative. It must be an absolute path. Run `realpath` on the `.pth` file to get the correct value.

**No detections even with people in view**  
Check `/scan` is being published (`ros2 topic hz /scan`). If the scan frequency is very low, increase `max_age` in the tracker to avoid tracks being dropped immediately.

**Decider node is not publishing `/cmd_vel`**  
Confirm that:
- `/goal_pose` has been published at least once
- `command_listener` is running and mode has been set to `2` or `3`
- `/tracked_objects_json` and `/predictions_json` are active (`ros2 topic hz` to check)

**TF lookup errors in the decider**  
The decider needs `base_link` → `odom` transforms. Verify your robot's TF tree with `ros2 run tf2_tools view_frames`.

**High latency or dropped-frame warnings**  
Run the frequency monitor and inspect the CSV output. If `dr_spaam` is the bottleneck, increase `stride`. If the predictor is slow, check GPU utilization.

---

## Citation

```bibtex
@inproceedings{yao2025towards,
    title={Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling},
    author={Yao, Jianpeng and Zhang, Xiaopan and Xia, Yu and Roy-Chowdhury, Amit K and Li, Jiachen},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2025}
}
```

## Acknowledgements

[DR-SPAAM](https://github.com/VisualComputingInstitute/DR-SPAAM-Detector) · [Gumbel Social Transformer](https://sites.google.com/view/gumbel-social-transformer) · [DtACI](https://github.com/isgibbs/DtACI) · [CrowdNav++](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph) · [OmniSafe](https://github.com/PKU-Alignment/omnisafe)
