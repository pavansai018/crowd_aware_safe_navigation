#!/usr/bin/env python3

import json
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from collections import deque
import torch
import os
import pickle  # To load args.pickle
import numpy as np
from copy import deepcopy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from .dt_aci import DtACI
from .human import Human
import time

from std_srvs.srv import Empty  # Import Empty service

# Import the predictor interface from gst_updated
from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import CrowdNavPredInterfaceMultiEnv


class Predictor(Node):
    def __init__(self):
        super().__init__('predictor_node')

        # ============= Parameter Declarations =============
        self.declare_parameters(
            namespace='',
            parameters=[
                # Publisher for predicted trajectories (Visualization)
                ('publisher.predicted_trajectories.topic', '/predicted_trajectories_viz'),  # Existing topic
                ('publisher.predicted_trajectories.queue_size', 10),
                ('publisher.predicted_trajectories.latch', False),

                # Publisher for predicted trajectories with ACI uncertainties (New)
                ('publisher.predicted_trajectories_aci_viz.topic', '/predicted_trajectories_aci_viz'),
                ('publisher.predicted_trajectories_aci_viz.queue_size', 10),
                ('publisher.predicted_trajectories_aci_viz.latch', False),

                # Subscriber for tracked_objects_json
                ('subscriber.tracked_objects_json.topic', '/tracked_objects_json'),
                ('subscriber.tracked_objects_json.queue_size', 10),

                # Model parameters
                ('model_path', ''),  # Default is empty string
                ('history_length', 5),  # Number of past positions to store

                # Publisher for predictions JSON to decider node
                ('publisher.predictions_json.topic', '/predictions_json'),  # Existing publisher
                ('publisher.predictions_json.queue_size', 10),
                ('publisher.predictions_json.latch', False),
            ]
        )

        # ============= Read Parameters =============
        self._read_params()

        # ============= Publishers =============
        self.marker_array_pub = self.create_publisher(
            MarkerArray,
            self.publisher_predicted_trajectories_topic,
            self.publisher_predicted_trajectories_queue_size  # Queue size as integer
        )

        # New Publisher for ACI-based Predictions Visualization
        self.marker_array_aci_pub = self.create_publisher(
            MarkerArray,
            self.publisher_predicted_trajectories_aci_viz_topic,
            self.publisher_predicted_trajectories_aci_viz_queue_size  # Queue size as integer
        )

        # New Publisher for Predictions JSON to Decider Node
        self.predictions_json_pub = self.create_publisher(
            String,
            self.publisher_predictions_json_topic,
            self.publisher_predictions_json_queue_size  # Queue size as integer
        )

        # ============= Subscriptions =============
        self.tracked_objects_json_sub_ = self.create_subscription(
            String,
            self.subscriber_tracked_objects_json_topic,
            self.tracked_objects_json_callback,
            self.subscriber_tracked_objects_json_queue_size  # Queue size as integer
        )

        # ============= Initialize Predictor Model =============
        model_path_param = self.model_path

        if model_path_param:
            model_dir = model_path_param
        else:
            from ament_index_python.packages import get_package_share_directory
            predictor_share_dir = get_package_share_directory('predictor')
            model_dir = os.path.join(predictor_share_dir, 'model_weight')  # Base model directory

        checkpoint_dir = os.path.join(model_dir, 'checkpoint')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args_pickle_path = os.path.join(checkpoint_dir, 'args.pickle')

        if not os.path.isfile(args_pickle_path):
            self.get_logger().error(f"[Predictor] args.pickle not found at: {args_pickle_path}")
            self.model = None
        else:
            with open(args_pickle_path, 'rb') as f:
                self.args = pickle.load(f)
            self.get_logger().info("[Predictor] Successfully loaded args.pickle.")

            self.model = CrowdNavPredInterfaceMultiEnv(
                load_path=model_dir,
                device=device,
                config=self.args,
                num_env=1
            )
            self.get_logger().info("[Predictor] Successfully initialized the prediction model.")

        self.history_length = self.history_length

        self.reset()

    def _read_params(self):
        """Reads parameters from node."""
        # Publishers
        self.publisher_predicted_trajectories_topic = self.get_parameter(
            'publisher.predicted_trajectories.topic').value
        self.publisher_predicted_trajectories_queue_size = self.get_parameter(
            'publisher.predicted_trajectories.queue_size').value
        self.publisher_predicted_trajectories_latch = self.get_parameter(
            'publisher.predicted_trajectories.latch').value

        self.publisher_predicted_trajectories_aci_viz_topic = self.get_parameter(
            'publisher.predicted_trajectories_aci_viz.topic').value
        self.publisher_predicted_trajectories_aci_viz_queue_size = self.get_parameter(
            'publisher.predicted_trajectories_aci_viz.queue_size').value
        self.publisher_predicted_trajectories_aci_viz_latch = self.get_parameter(
            'publisher.predicted_trajectories_aci_viz.latch').value

        self.publisher_predictions_json_topic = self.get_parameter(
            'publisher.predictions_json.topic').value
        self.publisher_predictions_json_queue_size = self.get_parameter(
            'publisher.predictions_json.queue_size').value
        self.publisher_predictions_json_latch = self.get_parameter(
            'publisher.predictions_json.latch').value

        # Subscribers
        self.subscriber_tracked_objects_json_topic = self.get_parameter(
            'subscriber.tracked_objects_json.topic').value
        self.subscriber_tracked_objects_json_queue_size = self.get_parameter(
            'subscriber.tracked_objects_json.queue_size').value

        # Model parameters
        self.model_path = self.get_parameter('model_path').value
        self.history_length = self.get_parameter('history_length').value

    def reset(self):
        # Create the Human objects (IDs from 0..49)
        self.max_human_num = 50
        self.humans = [Human(id=i, x=15.0, y=15.0) for i in range(self.max_human_num)]
        for human in self.humans:
            human.reset_aci(alpha=0.1)
        self.step_counter = 0

        self.pred_interval = 1
        self.buffer_len = (self.args.obs_seq_len - 1) * self.pred_interval + 1

        self.num_envs = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Make sure to specify float32 for these buffers
        self.traj_buffer = deque(
            [
                -999.0 * torch.ones(
                    (self.num_envs, self.max_human_num, 2),
                    dtype=torch.float32,
                    device=self.device
                )
                for _ in range(self.buffer_len)
            ],
            maxlen=self.buffer_len
        )

        # Boolean mask buffer
        self.mask_buffer = deque(
            [
                torch.zeros(
                    (self.num_envs, self.max_human_num, 1),
                    dtype=torch.bool,
                    device=self.device
                )
                for _ in range(self.buffer_len)
            ],
            maxlen=self.buffer_len
        )

    def tracked_objects_json_callback(self, msg: String):
        """
        Callback function to process incoming JSON messages from /tracked_objects_json.
        Expected JSON structure:
        {
          "header": {
             "frame_id": "laser",
             "stamp_sec": 1733440988,
             "stamp_nsec": 246987811
          },
          "tracks": [
             {"id": 1, "x": 1.23, "y": 4.56, "timestamp": 1733440988.246987811},
             {"id": 2, "x": 7.89, "y": 0.12, "timestamp": 1733440988.246987811},
             ...
          ]
        }
        """
        # Clear old markers before publishing new ones
        self.clear_old_markers(msg.data)

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            # self.get_logger().error("[Predictor] Received malformed JSON.")
            return

        tracks = data.get("tracks", [])
        header = data.get("header", {})
        frame_id = header.get("frame_id", "odom")  # Default to 'laser' if not specified
        stamp_sec = header.get("stamp_sec", 0)
        stamp_nsec = header.get("stamp_nsec", 0)

        if not tracks:
            # self.get_logger().info("[Predictor] Received '/tracked_objects_json' but it's empty.")
            return

        self.step_counter += 1
        # self.get_logger().info(f"[Predictor] Received JSON for {len(tracks)} tracked agents:")

        # Create a boolean mask for which humans are visible in the current step.
        self.visible_mask = torch.zeros(
            (self.max_human_num,),
            dtype=torch.bool,
            device=self.device
        )

        # Prepare MarkerArrays to collect all markers
        markers = MarkerArray()
        markers_aci = MarkerArray()  # For ACI visualization

        centers = []

        for agent in tracks:
            agent_id = agent.get("id", -1)
            orig_x = agent.get("x", 15.0)
            orig_y = agent.get("y", 15.0)
            t = agent.get("timestamp", 0.0)

            # --- Optional shift if your data is 1-based indexing ---
            # If your data has IDs 1..50, uncomment below to shift:
            # agent_id = agent_id - 1

            # Check the range to avoid out-of-bounds
            if agent_id < 0 or agent_id >= self.max_human_num:
                self.get_logger().warning(
                    f"[Predictor] Skipping agent with out-of-range ID={agent_id} "
                    f"(position=({orig_x:.2f},{orig_y:.2f}))"
                )
                continue

            x, y = orig_x, orig_y

            # Update the Human object's position
            self.humans[agent_id].set_attributes(x, y, t)

            # Mark this agent as visible
            self.visible_mask[agent_id] = True

        # Visualize current positions for visible humans
        num_visualization_humans = 0
        centers = []
        for human in self.humans:
            # Check if this human was visible
            if not self.visible_mask[human.id]:
                continue
            current_marker = self.create_current_position_marker(
                human.id,
                human.get_position(),
                frame_id=frame_id
            )
            markers.markers.append(current_marker)
            centers.append(human.get_position())
            num_visualization_humans += 1

        # self.get_logger().info(f"[Predictor] Visualized {num_visualization_humans} current positions.")
        # self.get_logger().info(f"[Predictor] Centers: {centers}")

        # Stack current positions for the model
        human_positions = np.array([h.get_position() for h in self.humans], dtype=np.float32)
        human_pos = torch.tensor(human_positions, device=self.device)
        # shape: [50, 2]
        human_pos = human_pos.unsqueeze(1)  # -> [50, 1, 2]
        human_pos = human_pos.transpose(0, 1)  # -> [1, 50, 2]

        # Append to the buffers
        self.traj_buffer.append(human_pos)  # float32
        # We expand visible_mask to shape [1, 50, 1]
        self.mask_buffer.append(self.visible_mask.unsqueeze(-1).unsqueeze(0))

        # Build the input tensors for the model
        in_traj = torch.stack(list(self.traj_buffer), dim=0).permute(1, 2, 0, 3)  # [buf_len, 1, 50, 2] -> [1, 50, buf_len, 2]
        in_mask = torch.stack(list(self.mask_buffer), dim=0).permute(1, 2, 0, 3)  # [1, 50, buf_len, 1]

        # Take every self.pred_interval element along the time dimension
        in_traj = in_traj[:, :, ::self.pred_interval]
        in_mask = in_mask[:, :, ::self.pred_interval].float()  # convert bool -> float for model

        if self.model is not None:
            gst_start_time = time.time()
            out_traj, out_mask = self.model.forward(
                input_traj=in_traj,
                input_binary_mask=in_mask
            )
            self.get_logger().info(f"[Predictor] Time for GST inference: {time.time() - gst_start_time}") 

            out_mask = out_mask.bool()  # revert to boolean mask

            # Store model outputs in each human object
            aci_start_time = time.time()
            for i, human in enumerate(self.humans):
                human.store_predictions(
                    out_traj[0, i, :, :2].cpu().numpy(),  # env index=0
                    is_valid=out_mask[0, i, :].cpu().numpy()
                )
                # logging the shape of out_traj and out_mask
                # self.get_logger().info(f"[Predictor] out_traj shape: {out_traj.shape}, out_mask shape: {out_mask.shape}") # [1, 50, 5, 2], [1, 50, 5]

                if self.visible_mask[human.id] and human.past_prediction_validity[-1]:
                    human.predictions_aci.append(deepcopy(human.past_predictions[-1]))
                    human.gt_locations_aci.append(human.get_position())
                    human.update_aci()
                    for j, aci_predictor in enumerate(human.pred_error_aci_list):
                        human.last_aci_predicted_conformity_score[j] = np.clip(aci_predictor.make_prediction(), 0, 1)
                    # logging human.predictions_aci for debugging
                    # self.get_logger().info(f"[Predictor] Human {human.id} predictions_aci: {human.predictions_aci[-1]}")        
            self.get_logger().info(f"[Predictor] Time for updating ACI: {time.time() - aci_start_time}") 
            # Visualize the predictions for all visible humans
            num_visualization = 0
            for human in self.humans:
                if not self.visible_mask[human.id]:
                    continue
                # Get the latest prediction from the buffer
                trajectory = list(human.past_predictions[-1])
                trajectory_marker = self.create_trajectory_marker(human.id, trajectory, frame_id=frame_id)
                markers.markers.append(trajectory_marker)
                num_visualization += 1

                # === New: Create and add ACI-based uncertainty markers ===
                # Ensure there are ACI scores available
                if len(human.last_aci_predicted_conformity_score) == 5:
                    aci_markers = self.create_aci_marker(
                        human.id,
                        trajectory,
                        human.last_aci_predicted_conformity_score,
                        frame_id=frame_id
                    )
                    markers_aci.markers.extend(aci_markers.markers)
                else:
                    self.get_logger().warning(
                        f"[Predictor] Human {human.id} has mismatched trajectory and ACI scores."
                    )

            # self.get_logger().info(f"[Predictor] Visualized {num_visualization} predicted trajectories.")

            # Build and publish predictions JSON to decider node
            predictions_json = self.build_predictions_json(header, self.humans)
            predictions_msg = String()
            predictions_msg.data = predictions_json
            self.predictions_json_pub.publish(predictions_msg)
            # self.get_logger().info(f"[Predictor] Published predictions JSON to '{self.publisher_predictions_json_topic}'.")

            # Publish ACI markers if any
            if markers_aci.markers:
                self.marker_array_aci_pub.publish(markers_aci)
                # self.get_logger().info(f"[Predictor] Published ACI uncertainty markers to '{self.publisher_predicted_trajectories_aci_viz_topic}'.")
        else:
            self.get_logger().warn("[Predictor] Model is None; skipping prediction.")

        # Publish trajectory markers
        self.marker_array_pub.publish(markers)

    def clear_old_markers(self, json_data):
        """
        Publishes a DELETEALL Marker to clear previously displayed markers.
        Extracts header information from the incoming JSON data.
        """
        try:
            data = json.loads(json_data)
            header = data.get("header", {})
            frame_id = header.get("frame_id", "odom")
            stamp_sec = header.get("stamp_sec", 0)
            stamp_nsec = header.get("stamp_nsec", 0)
        except json.JSONDecodeError:
            self.get_logger().error("[Predictor] Failed to decode JSON for header information.")
            return

        # Clear trajectory markers
        clear_msg = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp.sec = stamp_sec
        clear_marker.header.stamp.nanosec = stamp_nsec
        clear_marker.ns = "predictor"  # Namespace should match the markers being published
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        clear_msg.markers.append(clear_marker)
        self.marker_array_pub.publish(clear_msg)

        # Also clear ACI markers
        clear_marker_aci = Marker()
        clear_marker_aci.header.frame_id = frame_id
        clear_marker_aci.header.stamp.sec = stamp_sec
        clear_marker_aci.header.stamp.nanosec = stamp_nsec
        clear_marker_aci.ns = "predictor_aci"  # Namespace for ACI markers
        clear_marker_aci.id = 0
        clear_marker_aci.action = Marker.DELETEALL
        clear_msg_aci = MarkerArray()
        clear_msg_aci.markers.append(clear_marker_aci)
        self.marker_array_aci_pub.publish(clear_msg_aci)

    def build_predictions_json(self, header, humans):
        """
        Constructs a JSON string containing predictions for each tracked agent.
        """
        predictions = []
        for human in humans:
            if not self.visible_mask[human.id]:
                continue
            # Assuming 'past_predictions' holds the predicted trajectories
            trajectory = [
                {"x": float(pos[0]), "y": float(pos[1])}
                for pos in human.past_predictions[-1]  # Latest prediction
            ]
            uncertainty = [float(i) for i in human.last_aci_predicted_conformity_score]
            predictions.append({
                "id": human.id,
                "predicted_trajectory": trajectory,
                "uncertainty": uncertainty
            })
            # self.get_logger().info(f"[Predictor] Human {human.id} predictions: {predictions}")
        data_dict = {
            "header": {
                "frame_id": header.get("frame_id", "laser"),
                "stamp_sec": header.get("stamp_sec", 0),
                "stamp_nsec": header.get("stamp_nsec", 0)
            },
            "predictions": predictions
        }

        json_str = json.dumps(data_dict, ensure_ascii=False)
        return json_str

    def create_trajectory_marker(self, agent_id, trajectory, frame_id='laser'):
        """
        Creates a Marker message for the predicted trajectory of an agent
        as dots (SPHERE_LIST) in the specified frame with blue color.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "predictor"
        marker.id = agent_id + 1000  # Offset ID to differentiate from position markers

        marker.type = Marker.SPHERE_LIST  # Changed from LINE_STRIP to SPHERE_LIST for dots
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5  # Reduced scale for better visualization
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        # Set color to blue
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.4)  # Blue with higher alpha

        for pos in trajectory:
            p = Point()
            p.x = float(pos[0])
            p.y = float(pos[1])
            p.z = 0.0
            marker.points.append(p)

        return marker

    def create_aci_marker(self, agent_id, trajectory, aci_scores, frame_id='laser'):
        """
        Creates a MarkerArray message for the predicted trajectory of an agent
        with ACI uncertainties.
        - Adds an extra sphere (0.25 m radius) at the current position.
        - Adds spheres around predicted positions with radii based on ACI scores.
          - Steps 1-2: Light blue, alpha=0.6
          - Steps 3-5: Light blue, alpha=0.3
        """
        marker_array = MarkerArray()

        # # sphere around current position
        # current_pos = trajectory[0]
        # extra_sphere = Marker()
        # extra_sphere.header.frame_id = frame_id
        # extra_sphere.header.stamp = self.get_clock().now().to_msg()
        # extra_sphere.ns = "predictor_aci"
        # extra_sphere.id = agent_id * 2000  # Unique ID per agent
        # extra_sphere.type = Marker.SPHERE
        # extra_sphere.action = Marker.ADD

        # extra_sphere.pose.position.x = float(current_pos[0])
        # extra_sphere.pose.position.y = float(current_pos[1])
        # extra_sphere.pose.position.z = 0.0

        # # === Modified: Deeper blue color for current position in ACI visualization ===
        # extra_sphere.color = ColorRGBA(r=0.0, g=0.4, b=0.8, a=0.6)  # Deeper blue
        # extra_sphere.scale.x = 0.5  # 0.25 m radius
        # extra_sphere.scale.y = 0.5
        # extra_sphere.scale.z = 0.5

        # marker_array.markers.append(extra_sphere)


        # sphere around current position
        current_pos = trajectory[0]
        extra_sphere = Marker()
        extra_sphere.header.frame_id = frame_id
        extra_sphere.header.stamp = self.get_clock().now().to_msg()
        extra_sphere.ns = f"predictor_aci"
        extra_sphere.id = agent_id * 4000  # Unique ID per agent
        extra_sphere.type = Marker.SPHERE
        extra_sphere.action = Marker.ADD

        extra_sphere.pose.position.x = float(current_pos[0])
        extra_sphere.pose.position.y = float(current_pos[1])
        extra_sphere.pose.position.z = 0.0

        # === Modified: Deeper blue color for current position in ACI visualization ===
        extra_sphere.color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=0.4)  # Deeper blue
        extra_sphere.scale.x = 0.75  # 0.25 m radius
        extra_sphere.scale.y = 0.75
        extra_sphere.scale.z = 0.1

        marker_array.markers.append(extra_sphere)

        # Iterate over future trajectory points and corresponding ACI scores
        # Skip the first point as it is the current position
        for idx, (pos, aci_score) in enumerate(zip(trajectory[1:], aci_scores)):
            uncertainty_marker = Marker()
            uncertainty_marker.header.frame_id = frame_id
            uncertainty_marker.header.stamp = self.get_clock().now().to_msg()
            uncertainty_marker.ns = f"predictor_aci"
            uncertainty_marker.id = agent_id * 100 + idx  # Unique ID per agent and step

            uncertainty_marker.type = Marker.SPHERE
            uncertainty_marker.action = Marker.ADD

            uncertainty_marker.pose.position.x = float(pos[0])
            uncertainty_marker.pose.position.y = float(pos[1])
            uncertainty_marker.pose.position.z = 0.0

            # Determine color and transparency based on prediction step
            if idx < 2:
                # Steps 1-2
                color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=0.4)  # Light blue with alpha 0.6
            else:
                # Steps 3-5
                color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=0.25)  # Light blue with alpha 0.3

            uncertainty_marker.color = color

            # Scale based on ACI score
            # For visualization, you can define a base scale and add scaling based on ACI score
            base_scale = 0.5  # Base radius
            scaled_scale = base_scale + (1.0 * aci_score)  # Example scaling
            uncertainty_marker.scale.x = scaled_scale
            uncertainty_marker.scale.y = scaled_scale
            uncertainty_marker.scale.z = scaled_scale / 2

            marker_array.markers.append(uncertainty_marker)

        return marker_array

    def create_current_position_marker(self, agent_id, position, frame_id='laser'):
        """
        Creates a Marker message for the current position of an agent
        in the specified frame.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "predictor"
        marker.id = agent_id  # Unique ID for each agent's current position

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.0

        marker.color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=0.8)  # Red
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        return marker

    def id_to_color(self, agent_id):
        """
        Generates a unique color for each agent based on their ID.
        """
        np.random.seed(agent_id)
        return np.random.rand(1).tolist()


def main(args=None):
    rclpy.init(args=args)
    node = Predictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Predictor] KeyboardInterrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
