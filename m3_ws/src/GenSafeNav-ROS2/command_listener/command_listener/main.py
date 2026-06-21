#!/usr/bin/env python3

import sys
import threading
import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import tty
import termios
import select

class CommandListenerNode(Node):
    def __init__(self):
        super().__init__('command_listener')
        
        # Publisher for sending commands. Adjust topic name / QoS as needed.
        self.command_publisher_ = self.create_publisher(String, '/command', 10)
        
        self.running_ = True

        # We’ll use a separate thread to handle user input, so the node can still be spun if needed.
        self.input_thread_ = threading.Thread(target=self.user_input_loop, daemon=True)
        self.input_thread_.start()

    def user_input_loop(self):
        """
        This loop continuously handles user prompts for mode selection and processes the selected mode.
        """
        while self.running_:
            # 1. Ask for the mode
            self.current_mode_ = None
            while self.current_mode_ not in [1, 2, 3] and self.running_:
                print("\nWhich mode do you want to choose? (1) Manual (2) Automatic (3) Combined mode")
                mode_str = self.blocking_input("Enter 1, 2, or 3: ")
                try:
                    mode = int(mode_str)
                    if mode in [1, 2, 3]:
                        self.current_mode_ = mode
                        self.publish_mode_command(mode)
                    else:
                        print("Invalid mode. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter 1, 2, or 3.")
            
            if not self.running_:
                break

            # 2. Handle the selected mode
            if self.current_mode_ == 1:
                self.listen_manual_input()
            elif self.current_mode_ in [2, 3]:
                self.decide_auto_or_combined()

    def publish_mode_command(self, mode: int):
        """Publishes the chosen mode to the /command topic."""
        msg = String()
        if mode == 1:
            msg.data = "mode:manual"
        elif mode == 2:
            msg.data = "mode:automatic"
        elif mode == 3:
            # Combined mode allows manual override during autonomous navigation,
            # enabling human intervention in emergency situations.
            msg.data = "mode:combined"
        self.command_publisher_.publish(msg)
        self.get_logger().info(f"Published mode command: {msg.data}")

    def listen_manual_input(self):
        """Continuously listens to WASD inputs, echoes them, and publishes corresponding commands until exit."""
        print("\nListening to manual input .... (Press 'q' to quit back to mode selection)")
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running_:
                if self.kbhit():
                    key = sys.stdin.read(1).lower()
                    
                    # Echo the key press
                    if key != '\x03':  # Ignore Ctrl+C
                        print(key, end='', flush=True)
                    
                    if key == 'q':
                        # MODIFIED: If we are in combined mode, publish "stop" as we exit
                        if self.current_mode_ == 3:
                            self.publish_stop_command()

                        print("\nExiting manual input to mode selection.")
                        break  # Exit manual input

                    elif key in ['w', 'a', 's', 'd']:
                        msg = String()
                        msg.data = f"manual:{key}"
                        self.command_publisher_.publish(msg)
                        self.get_logger().debug(f"Published manual command: {msg.data}")
                    elif key == '\x03':
                        print("\nKeyboard interrupt received. Exiting.")
                        self.running_ = False
                        break
                    else:
                        print("\nInvalid key. Use W/A/S/D or Q to quit.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def decide_auto_or_combined(self):
        """
        After choosing automatic or combined mode, ask user:
        1) a goal position
        2) moving forward
        Then publish the corresponding command.

        If combined mode, allows manual overrides.
        """
        prompt = (
            "\nDo you want (1) a goal position or (2) moving forward or (3) RViz goal?\n"
            "Enter 1, 2, or 3: "
        )
        sub_choice = None
        while sub_choice not in [1, 2, 3] and self.running_:
            try:
                sub_choice_str = self.blocking_input(prompt)
                sub_choice = int(sub_choice_str)
            except ValueError:
                print("Invalid choice. Please enter 1, 2, or 3.")
                continue

        if not self.running_:
            return

        if sub_choice == 1:
            # Ask for x,y
            xy_input = self.blocking_input("Enter goal position as 'x,y': ")
            try:
                x_str, y_str = xy_input.split(',')
                x_val = float(x_str.strip())
                y_val = float(y_str.strip())
                self.publish_goal_command(x_val, y_val)
            except Exception as e:
                print(f"Invalid input for x,y. Error: {e}")
                return
        elif sub_choice == 2:
            # Moving forward
            self.publish_forward_command()
        else:
            # RViz goal — just set automatic mode, user clicks goal in RViz
            print("\nClick '2D Nav Goal' in RViz to set the goal. Press 'q' to stop.")
        
        if self.current_mode_ == 3:
            print("\nCombined mode is active. The decider node is running by default.")
            print("Manual WASD commands will override the decider if pressed.")
            self.listen_manual_input()
        elif self.current_mode_ == 2:
            print("\nAutomatic mode is active. Press 'q' to quit to mode selection.")
            self.listen_quit_input()  # Only 'q' now

    def publish_goal_command(self, x: float, y: float):
        msg = String()
        msg.data = f"auto:goal:{x},{y}"
        self.command_publisher_.publish(msg)
        self.get_logger().info(f"Published goal command: {msg.data}")

    def publish_forward_command(self):
        msg = String()
        msg.data = "auto:forward"
        self.command_publisher_.publish(msg)
        self.get_logger().info(f"Published forward command: {msg.data}")

    def publish_stop_command(self):
        """Publishes a stop command to the /command topic."""
        msg = String()
        msg.data = "stop"
        self.command_publisher_.publish(msg)
        self.get_logger().info(f"Published stop command: {msg.data}")

    # REMOVED: publish_quit_command (and references to it)
    # def publish_quit_command(self):
    #     """Publishes a 'quit' command to the /command topic."""
    #     pass

    def listen_quit_input(self):
        """Listen for a key to quit current mode back to mode selection."""
        print("Press 'q' to return to mode selection.")
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running_:
                if self.kbhit():
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'q':
                        # Return to mode selection and stop
                        print("\nExiting to mode selection.")
                        self.publish_stop_command()
                        break
                    else:
                        print("\nInvalid key. Press 'q'.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def kbhit(self):
        """Check if a key has been pressed."""
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return dr != []

    def blocking_input(self, prompt=""):
        """
        Displays a prompt and reads input without interfering with raw mode.
        This function is used only for initial mode selection and sub-mode selection,
        where line-buffered input is acceptable.
        """
        print(prompt, end='', flush=True)
        return sys.stdin.readline().strip()


def main(args=None):
    rclpy.init(args=args)
    node = CommandListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        node.running_ = False
        if node.input_thread_.is_alive():
            node.input_thread_.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
