#!/usr/bin/env python3

import json
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

import tf2_ros
import tf_transformations
import numpy as np
import os
import torch
import torch.nn as nn
import time

import gym

from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

from rl.networks.model import Policy
from ament_index_python.packages import get_package_share_directory

MANUAL_CMD_TIMEOUT = 0.2  # 200 ms override window
MAX_HUMANS = 50

class Decider(Node):
    def __init__(self):
        super().__init__('decider_node')

        # ============= Subscriptions =============
        self.command_sub_ = self.create_subscription(
            String,
            '/command',
            self.command_callback,
            10
        )
        self.joint_state_sub_ = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.tracked_objects_sub_ = self.create_subscription(
            String,
            '/tracked_objects_json',
            self.tracked_objects_json_callback,
            10
        )
        self.predictions_sub_ = self.create_subscription(
            String,
            '/predictions_json',
            self.predictions_callback,
            10
        )
        self.odom_sub_ = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            20
        )
        self.goal_pose_sub_ = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )

        # ============= TF2 for Robot Pose =============
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        # ============= Publishers =============
        self.cmd_vel_pub_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.action_marker_pub_ = self.create_publisher(Marker, '/decider_action_marker', 10)

        # NEW: A dedicated publisher for the "goal" marker
        self.goal_marker_pub_ = self.create_publisher(Marker, '/decider_goal_marker', 10)

        # NEW: A dedicated publisher for the robot position marker (orange circle)
        self.robot_marker_pub_ = self.create_publisher(Marker, '/decider_robot_marker', 10)

        # ============= Mode and override =============
        self.current_mode_ = None  # "manual", "automatic", or "combined"
        self.in_override_ = False
        self.last_manual_cmd_time_ = self.get_clock().now()
        self.timer_ = self.create_timer(0.1, self.check_manual_timeout)

        # ============= RL Setup =============
        from config.arguments import get_args
        self.algo_args = get_args()
        self.get_logger().info("[Decider] Successfully imported algo_args.")

        from config.config import Config
        self.config = Config()
        self.get_logger().info("[Decider] Successfully imported config.")

        self.predict_steps = 5
        self.set_ob_act_space()

        self.actor_critic = Policy(
            self.observation_space,
            self.action_space,
            self.config,
            base_kwargs=self.algo_args,
            base='selfAttn_merge_srnn'
        )

        decider_share_dir = get_package_share_directory('decider')
        load_path = os.path.join(decider_share_dir, 'model_weight','ours.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic.load_state_dict(torch.load(load_path, map_location=self.device))
        self.actor_critic.base.nenv = 1
        nn.DataParallel(self.actor_critic).to(self.device)
        self.get_logger().info("[Decider] RL model initialized.")

        # Recurrent states
        self.eval_recurrent_hidden_states = {}
        self.eval_masks = torch.zeros(1, 1, device=self.device)

        # Robot data
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_vpref = 1.0
        self.robot_radius = 0.3
        self.robot_vx = 0.0
        self.robot_vy = 0.0

        # Movement/goal logic
        self.forward_mode = False
        self.goal_x = None
        self.goal_y = None
        self.last_known_pose = (0.0, 0.0, 0.0)

        # Tracked data
        self.current_predictions_ = None
        self.tracked_humans = []
        self.detect_range = 5.0

        # ============== Smoothing / Clipping ==============
        self.motion_mode = "mecanum"
        self.max_delta_vx = 0.25   # max change in vx each step
        self.max_delta_vy = 0.25   # max change in vy each step
        self.max_speed = 0.7      # absolute max speed
        self.last_rl_vel = np.array([0.0, 0.0], dtype=np.float32)

    def set_ob_act_space(self):
        import gym
        d = {}
        d['robot_node'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,7), dtype=np.float32)
        d['temporal_edges'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,2), dtype=np.float32)

        self.spatial_edge_dim = int(2*(self.predict_steps+1))
        d['spatial_edges'] = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(self.config.sim.human_num + self.config.sim.human_num_range, self.spatial_edge_dim),
            dtype=np.float32
        )

        # d['visible_masks'] = gym.spaces.Box(
        #     -np.inf, np.inf,
        #     shape=(self.config.sim.human_num + self.config.sim.human_num_range,),
        #     dtype=np.bool_
        # )
        d['visible_masks'] = gym.spaces.Box(low=0, high=1, shape=(self.config.sim.human_num + self.config.sim.human_num_range,), dtype=np.bool_,)
        d['detected_human_num'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        d['aggressiveness_factor'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,1), dtype=np.float32)
        d['conformity_scores'] = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(self.config.sim.human_num + self.config.sim.human_num_range, self.predict_steps),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(d)
        high = np.inf * np.ones([2,])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    # ============ Timer / Manual Timeout ============
    def check_manual_timeout(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_manual_cmd_time_).nanoseconds * 1e-9

        if self.current_mode_ == "manual":
            if elapsed > MANUAL_CMD_TIMEOUT:
                self.publish_stop_command()
                self.get_logger().debug("[Decider] Manual => timed out => stopping.")

        elif self.current_mode_ == "combined":
            if self.in_override_ and elapsed > MANUAL_CMD_TIMEOUT:
                self.in_override_ = False
                self.get_logger().info("[Decider] Combined override timed out => revert to RL.")
                self.publish_stop_command()

    def rl_reset(self):
        # Full RL reset, including goals:
        self.eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(
            1, 1,
            self.actor_critic.base.human_node_rnn_size,
            device=self.device
        )
        self.eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(
            1, self.actor_critic.base.human_num+1,
            self.actor_critic.base.human_human_edge_rnn_size,
            device=self.device
        )
        self.eval_masks = torch.zeros(1,1, device=self.device)
        self.goal_x = None
        self.goal_y = None
        self.forward_mode = False
        self.last_rl_vel[:] = 0.0
        self.get_logger().info("[Decider] RL reset complete.")

    # (MOD) Add a helper function to reset only the recurrent states
    def reset_recurrent_states(self):
        self.eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(
            1, 1,
            self.actor_critic.base.human_node_rnn_size,
            device=self.device
        )
        self.eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(
            1, self.actor_critic.base.human_num+1,
            self.actor_critic.base.human_human_edge_rnn_size,
            device=self.device
        )
        self.eval_masks = torch.zeros(1,1, device=self.device)
        self.last_rl_vel[:] = 0.0
        self.get_logger().warn("[Decider] Recurrent states reset due to NaN or exception.")

    def _run_simple_goal_controller(self):
        rx, ry, rtheta = self.robot_x, self.robot_y, self.robot_theta
        if self.forward_mode:
            gx, gy = rx + 5.0, 0.0
        elif self.goal_x is not None and self.goal_y is not None:
            gx, gy = self.goal_x, self.goal_y
        else:
            return

        dx, dy = gx - rx, gy - ry
        dist = np.hypot(dx, dy)
        if dist < 0.1:
            self.publish_stop_command()
            return

        vx = (dx / dist) * self.robot_vpref
        vy = (dy / dist) * self.robot_vpref

        vx_smooth, vy_smooth = self.smooth_and_clip_mecanum(
            np.array([vx, vy], dtype=np.float32), self.last_rl_vel
        )
        self.last_rl_vel[:] = [vx_smooth, vy_smooth]
        self.publish_velocity_command_global(vx_smooth, vy_smooth, require_tf_for_rl=True)
        self.publish_goal_marker()

    # ============ Subscriptions ============
    def odom_callback(self, msg: Odometry):
        # Store linear velocities
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_vy = msg.twist.twist.linear.y

        # Also get the position/heading from Odometry if needed
        # (However, you might be using TF for more accurate robot pose.)
        # For marker visualization, we can simply use the Odom's position:
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Orientation from Odometry
        rot = msg.pose.pose.orientation
        q = (rot.x, rot.y, rot.z, rot.w)
        _, _, yaw = tf_transformations.euler_from_quaternion(q)
        self.robot_theta = yaw

        # Publish the orange circle marker for the robot at every odom update
        self.publish_robot_marker()

    def goal_pose_callback(self, msg: PoseStamped):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.forward_mode = False
        self.current_mode_ = "automatic"
        self.get_logger().info(f"[Decider] RViz goal => ({self.goal_x:.2f}, {self.goal_y:.2f})")

    def joint_state_callback(self, msg: JointState):
        pass

    def tracked_objects_json_callback(self, msg: String):
        # Only do RL if in auto/combined mode
        if self.current_mode_ not in ("automatic", "combined"):
            return

        # (MOD) If in combined mode but currently overridden by manual, skip RL
        if self.current_mode_ == "combined" and self.in_override_:
            return

        # If user hasn't chosen forward or a (goal_x, goal_y), skip RL
        if not self.forward_mode and (self.goal_x is None or self.goal_y is None):
            return

        try:
            data = json.loads(msg.data)
            tracks = data.get("tracks", [])
            if not tracks:
                return

            # We need a valid TF
            robot_pose = self.get_robot_pose(require_tf_for_rl=True)
            if robot_pose is None:
                self.get_logger().warn("[Decider] No TF => skip RL step.")
                return

            self.robot_x, self.robot_y, self.robot_theta = robot_pose

            # Filter / sort humans
            tmp = []
            for agent in tracks:
                hid = agent.get("id", -1)
                if hid < 0:
                    continue
                hx = agent.get("x", 15.0)
                hy = agent.get("y", 15.0)
                dist = np.hypot(hx - self.robot_x, hy - self.robot_y)
                if dist <= self.detect_range:
                    tmp.append((hid, hx, hy, dist))
            tmp.sort(key=lambda x: x[3])
            self.tracked_humans = tmp

            # No humans in range — use simple go-to-goal instead of RL
            # if not self.tracked_humans:
            #     self._run_simple_goal_controller()
            #     return

            # Check if goal reached
            if not self.forward_mode and self.goal_x is not None:
                dist_to_goal = np.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
                if dist_to_goal < 0.3:
                    self.publish_stop_command()
                    return

            # RL step
            obs = self.build_observation_from_humans()
            self.run_rl_step(obs)

        except json.JSONDecodeError as ex:
            self.get_logger().error(f"[Decider] JSON parse error in tracked_objects: {ex}")

    def predictions_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            predictions_list = data.get("predictions", [])
            if not predictions_list:
                return
            pred_dict = {}
            for hum in predictions_list:
                hid = hum["id"]
                pred_dict[hid] = {
                    "predicted_trajectory": hum["predicted_trajectory"],
                    "uncertainty": hum.get("uncertainty", [])
                }
            self.current_predictions_ = pred_dict
        except json.JSONDecodeError as ex:
            self.get_logger().error(f"[Decider] JSON parse error in predictions: {ex}")

    # ============ Observations ============
    def build_observation_from_humans(self):
        rx, ry, rtheta = self.robot_x, self.robot_y, self.robot_theta

        # For forward_mode, place the goal 5m ahead in x, y=0 in the global odom
        if self.forward_mode:
            # gx, gy = rx + 5.0, 0.0
            gx, gy = rx + 5.0 * np.cos(rtheta), ry + 5.0 * np.sin(rtheta)
        elif self.goal_x is not None and self.goal_y is not None:
            gx, gy = self.goal_x, self.goal_y
        else:
            gx, gy = rx, ry

        robot_node = np.array([[rx, ry, self.robot_radius, gx, gy, self.robot_vpref, rtheta]], dtype=np.float32)
        temporal_edges = np.array([[self.robot_vx, self.robot_vy]], dtype=np.float32)

        nHumansMax = 50
        spatial_edges = np.ones((nHumansMax, self.spatial_edge_dim), dtype=np.float32)*15
        visible_masks = np.zeros((nHumansMax,), dtype=bool)
        conformity_scores = np.zeros((nHumansMax, self.predict_steps), dtype=np.float32)

        sorted_humans = self.tracked_humans[:nHumansMax]
        detected_num = len(sorted_humans)

        for i,(hid,hx,hy,dist) in enumerate(sorted_humans):
            dx = hx - rx
            dy = hy - ry
            spatial_edges[i,0] = dx
            spatial_edges[i,1] = dy

            if self.current_predictions_ and (hid in self.current_predictions_):
                pred_traj = self.current_predictions_[hid].get("predicted_trajectory",[])
                for step_idx in range(self.predict_steps):
                    if step_idx < len(pred_traj):
                        px = pred_traj[step_idx].get("x",15.0) - rx
                        py = pred_traj[step_idx].get("y",15.0) - ry
                        idx2 = 2*(step_idx+1)
                        spatial_edges[i, idx2]   = px
                        spatial_edges[i, idx2+1] = py

                uncertain_list = self.current_predictions_[hid].get("uncertainty",[])
                for step_idx in range(self.predict_steps):
                    if step_idx < len(uncertain_list):
                        val = uncertain_list[step_idx]
                        val = max(0.0, min(1.0, val))
                        conformity_scores[i, step_idx] = val

            visible_masks[i] = True

        obs_dict = {
            "robot_node": robot_node,
            "temporal_edges": temporal_edges,
            "spatial_edges": spatial_edges,
            "visible_masks": visible_masks,
            "detected_human_num": np.array([detected_num], dtype=np.float32),
            "aggressiveness_factor": np.zeros((1,1), dtype=np.float32),
            "conformity_scores": conformity_scores
        }

        obs = {}
        for k,v in obs_dict.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1:
                    obs[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
                elif v.ndim == 2:
                    obs[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
                else:
                    obs[k] = torch.from_numpy(v).to(self.device)
        return obs

    # ============ RL Inference + Smoothing =============
    def run_rl_step(self, obs):
        # (MOD) Wrap RL inference in try/except
        try:
            with torch.no_grad():
                rl_start_time = time.time()
                _, action_tensor, _, self.eval_recurrent_hidden_states = self.actor_critic.act(
                    obs,
                    self.eval_recurrent_hidden_states,
                    self.eval_masks,
                    deterministic=True
                )
                rl_elapsed = time.time() - rl_start_time
                self.get_logger().info(f"[Decider] Time for RL inference: {rl_elapsed}") 


            # (MOD) Check for NaNs in the action tensor
            if torch.isnan(action_tensor).any():
                self.get_logger().warn("[Decider] NaN detected in action_tensor => resetting states.")
                self.reset_recurrent_states()
                return

            raw_action = action_tensor.cpu().numpy()[0]
            vx_raw, vy_raw = raw_action[0], raw_action[1]

            new_vel = np.array([vx_raw, vy_raw], dtype=np.float32)
            vx_smooth, vy_smooth = self.smooth_and_clip_mecanum(new_vel, self.last_rl_vel)
            self.last_rl_vel[:] = [vx_smooth, vy_smooth]

            # Publish RL velocity only if no manual override
            if not self.in_override_:
                self.publish_velocity_command_global(vx_smooth, vy_smooth, require_tf_for_rl=True)
            else:
                self.get_logger().debug("[Decider] in_override_ = True => skipping RL cmd publish.")

            # NEW: Publish the goal marker
            self.publish_goal_marker()

        except Exception as ex:
            self.get_logger().error(f"[Decider] RL step exception: {ex}")
            # Reset recurrent states to avoid stuck NaNs
            self.reset_recurrent_states()

    def smooth_and_clip_mecanum(self, new_vel, last_vel):
        vx_current, vy_current = last_vel
        vx_target, vy_target = new_vel

        delta_vx = np.clip(vx_target - vx_current, -self.max_delta_vx, self.max_delta_vx)
        delta_vy = np.clip(vy_target - vy_current, -self.max_delta_vy, self.max_delta_vy)

        vx_candidate = vx_current + delta_vx
        vy_candidate = vy_current + delta_vy

        speed = np.hypot(vx_candidate, vy_candidate)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx_candidate *= scale
            vy_candidate *= scale

        return float(vx_candidate), float(vy_candidate)

    # ============= Command Callback =============
    def command_callback(self, msg: String):
        cmd_str = msg.data
        self.get_logger().info(f"[Decider] Command: {cmd_str}")

        if cmd_str == "stop":
            self.publish_stop_command()
            self.current_mode_ = None
            return

        if cmd_str.startswith("mode:"):
            new_mode = cmd_str.split(":")[1]
            self.current_mode_ = new_mode
            self.get_logger().info(f"[Decider] Switched to mode: {self.current_mode_}")
            if new_mode in ("automatic", "combined"):
                self.rl_reset()
            else:
                self.in_override_ = False
                self.publish_stop_command()
            return

        if self.current_mode_ == "manual":
            if cmd_str.startswith("manual:"):
                self.handle_manual_command(cmd_str)
            return

        # Automatic or Combined
        if self.current_mode_ in ("automatic","combined"):
            if cmd_str == "auto:forward":
                self.forward_mode = True
                # Clear old goal
                self.goal_x = None
                self.goal_y = None
                self.get_logger().info("[Decider] auto:forward => RL step uses (rx+5, 0).")
            elif cmd_str.startswith("auto:goal:"):
                try:
                    s = cmd_str.split("auto:goal:")[-1]
                    gx_str, gy_str = s.split(",")
                    self.goal_x = float(gx_str)
                    self.goal_y = float(gy_str)
                    self.forward_mode = False
                    self.get_logger().info(f"[Decider] auto:goal => ({self.goal_x}, {self.goal_y})")
                except:
                    self.get_logger().error("[Decider] auto:goal parse error.")
            elif cmd_str.startswith("manual:") and self.current_mode_ == "combined":
                self.handle_manual_command(cmd_str)
            else:
                self.get_logger().warn(f"[Decider] Unknown command: {cmd_str}")

    def handle_manual_command(self, cmd_str: str):
        self.last_manual_cmd_time_ = self.get_clock().now()
        if self.current_mode_ == "combined":
            self.in_override_ = True

        key = cmd_str.split(":")[1]
        if key == 'w':
            vx_global, vy_global = 0.5, 0.0
        elif key == 's':
            vx_global, vy_global = -0.5, 0.0
        elif key == 'a':
            vx_global, vy_global = 0.0, 0.5
        elif key == 'd':
            vx_global, vy_global = 0.0, -0.5
        else:
            self.get_logger().warn(f"[Decider] Unrecognized manual key: {key}")
            return

        self.publish_velocity_command_global(vx_global, vy_global, require_tf_for_rl=False)

    # ============= Utility Functions =============

    # NEW: Publish a red sphere marker for the current "goal"
    def publish_goal_marker(self):
        """Publishes an orange cylinder in 'odom' frame at the current goal."""
        rx, ry, rtheta = self.robot_x, self.robot_y, self.robot_theta

        if self.forward_mode:
            gx, gy = rx + 5.0, 0.0
        elif self.goal_x is not None and self.goal_y is not None:
            gx, gy = self.goal_x, self.goal_y
        else:
            gx, gy = rx, ry  # fallback if no valid goal

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "decider_goal"
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Twice the robot's radius = 0.6 => diameter = 1.2
        diameter = 2.0 * (1.5 * self.robot_radius)  # 1.2 if robot_radius = 0.3
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = 0.02  # small height to make it a "filled circle"

        # Orange color
        marker.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=0.6)

        marker.pose.position.x = gx
        marker.pose.position.y = gy
        marker.pose.position.z = 0.0

        # Orientation identity
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.goal_marker_pub_.publish(marker)

    # NEW: Publish an orange circle marker for the robot
    def publish_robot_marker(self):
        """Publishes an orange circle (sphere) of radius 0.2 in 'odom' frame at the robot's position."""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "decider_robot"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Robot radius = 0.2 => diameter = 0.4
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4

        # Orange color
        marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)

        # Position from odom
        marker.pose.position.x = self.robot_x
        marker.pose.position.y = self.robot_y
        marker.pose.position.z = 0.0

        # Orientation identity
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.robot_marker_pub_.publish(marker)

    def publish_velocity_command_global(self, vx_global: float, vy_global: float, require_tf_for_rl: bool=False):
        pose = self.get_robot_pose(require_tf_for_rl)
        if pose is None and require_tf_for_rl:
            self.get_logger().warn("[Decider] RL velocity not published, no TF.")
            return

        if pose is not None:
            rx, ry, ryaw = pose
        else:
            rx, ry, ryaw = self.last_known_pose
            self.get_logger().warn(f"[Decider] Using last_known_pose={self.last_known_pose} for manual marker.")

        local_vx = np.cos(ryaw)*vx_global + np.sin(ryaw)*vy_global
        local_vy = -np.sin(ryaw)*vx_global + np.cos(ryaw)*vy_global

        twist_msg = Twist()
        twist_msg.linear.x = float(local_vx)
        twist_msg.linear.y = float(local_vy)
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub_.publish(twist_msg)

        self.get_logger().info(
            f"[Decider] => G=({vx_global:.2f},{vy_global:.2f}), L=({local_vx:.2f},{local_vy:.2f})"
        )

        # Publish arrow for velocity
        self.publish_action_marker_global(rx, ry, vx_global, vy_global)

    def publish_stop_command(self):
        twist_msg = Twist()
        self.cmd_vel_pub_.publish(twist_msg)
        self.get_logger().info("[Decider] Stop command.")

        rx, ry, ryaw = self.last_known_pose
        self.publish_action_marker_global(rx, ry, 0.0, 0.0)

    def get_robot_pose(self, require_tf_for_rl=False):
        try:
            # tf_stamped = self.tf_buffer_.lookup_transform('odom','base_link', rclpy.time.Time())
            tf_stamped = self.tf_buffer_.lookup_transform('map','base_link', rclpy.time.Time())
            trans = tf_stamped.transform.translation
            rot = tf_stamped.transform.rotation
            q = (rot.x, rot.y, rot.z, rot.w)
            _, _, yaw = tf_transformations.euler_from_quaternion(q)
            rx, ry = trans.x, trans.y
            self.last_known_pose = (rx, ry, yaw)
            return (rx, ry, yaw)
        except Exception as e:
            if require_tf_for_rl:
                self.get_logger().error(f"[Decider] TF lookup failed for RL: {e}")
                return None
            else:
                self.get_logger().warn("TF lookup failed for manual. Using last_known_pose.")
                return None

    def publish_action_marker_global(self, rx, ry, vx_g, vy_g):
        vel_mag = np.hypot(vx_g, vy_g)
        arrow_theta = np.arctan2(vy_g, vx_g) if vel_mag > 0.001 else 0.0

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "decider_action"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.scale.x = float(vel_mag)
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)

        marker.pose.position.x = rx
        marker.pose.position.y = ry
        q = tf_transformations.quaternion_from_euler(0, 0, arrow_theta)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        self.action_marker_pub_.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = Decider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Decider] KeyboardInterrupt -> shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
