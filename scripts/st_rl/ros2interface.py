import os, sys
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import torch

from rclpy.qos import QoSProfile, ReliabilityPolicy

#print(ReliabilityPolicy.__dict__)
#from rclpy.qos import qos_profile_default, qos_profile_sensor_data

class Ros2RealRobot(Node):

    def __init__(self,
            namespace= None,
            low_state_topic= "/state",
            low_cmd_topic= "/action",
            joy_stick_topic = "/joy_cmd",
            forward_depth_topic = None, # if None and still need access, set to str "pyrealsense"
            forward_depth_embedding_topic = "/forward_depth_embedding",
            cfg= dict(),
            lin_vel_deadband= 0.1,
            ang_vel_deadband= 0.1,
            replace_obs_with_embeddings= [], # a list of strings, e.g. ["forward_depth"] then the corrseponding obs will be processed by _get_forward_depth_embedding_obs()
            move_by_wireless_remote= True, # if True, the robot will be controlled by a wireless remote
            model_device= "cpu",
            dof_pos_protect_ratio= 1.05, # if the dof_pos is out of the range of this ratio, the process will shutdown.
            robot_class_name= "AmbotN1",
            dryrun= True, # if True, the robot will not send commands to the real robot
        ):
        super().__init__("ros2_rlcontroller_node")
        self.NUM_DOF = getattr(RobotCfgs, robot_class_name).NUM_DOF
        self.NUM_ACTIONS = getattr(RobotCfgs, robot_class_name).NUM_ACTIONS
        self.namespace = namespace

        self.low_state_topic = low_state_topic if namespace is None else namespace+low_state_topic
        self.low_cmd_topic = low_cmd_topic if namespace is None else namespace+low_cmd_topic   #if not dryrun else low_cmd_topic + "_dryrun_" + str(np.random.randint(0, 65535))
        self.joy_stick_topic = joy_stick_topic if namespace is None else namespace + joy_stick_topic
        self.forward_depth_topic = forward_depth_topic if namespace is None else (None if forward_depth_topic is None else namespace + forward_depth_topic)
        self.forward_depth_embedding_topic =  forward_depth_embedding_topic if namespace is None else (None if forward_depth_embedding_topic is None else namespace + forward_depth_embedding_topic)
        self.cfg = cfg

        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.replace_obs_with_embeddings = replace_obs_with_embeddings
        self.move_by_wireless_remote = move_by_wireless_remote
        self.model_device = model_device
        self.dof_pos_protect_ratio = dof_pos_protect_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun

        self.dof_map = getattr(RobotCfgs, robot_class_name).dof_map
        self.dof_names = getattr(RobotCfgs, robot_class_name).dof_names
        self.dof_signs = getattr(RobotCfgs, robot_class_name).dof_signs
        self.turn_on_motor_mode = getattr(RobotCfgs, robot_class_name).turn_on_motor_mode

        self.parse_config()

    def parse_config(self):
        """ parse, set attributes from config dict, initialize buffers to speed up the computation """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = torch.zeros((1, 3), device= self.model_device, dtype= torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1
        
        # observations
        self.clip_obs = self.cfg["normalization"]["clip_observations"]
        self.obs_scales = self.cfg["normalization"]["obs_scales"]
        for k, v in self.obs_scales.items():
            if isinstance(v, (list, tuple)):
                self.obs_scales[k] = torch.tensor(v, device= self.model_device, dtype= torch.float32)
        # check whether there are embeddings in obs_components and launch encoder process later
        if len(self.replace_obs_with_embeddings) > 0:
            for comp in self.replace_obs_with_embeddings:
                self.get_logger().warn(f"{comp} will be replaced with its embedding when get_obs, don't forget to launch the corresponding process before running the policy.")
        self.obs_segments = self.get_obs_segment_from_components(self.cfg["env"]["obs_components"])
        self.num_obs = self.get_num_obs_from_components(self.cfg["env"]["obs_components"])

        if "privileged_obs_components" in self.cfg["env"].keys():
            self.privileged_obs_segments = self.get_obs_segment_from_components(self.cfg["env"]["privileged_obs_components"])
            self.num_privileged_obs = self.get_num_obs_from_components(self.cfg["env"]["privileged_obs_components"])
        else:
            self.privileged_obs_segments = self.obs_segments
            self.num_privileged_obs = self.num_obs

        for obs_component in self.cfg["env"]["obs_components"]:
            if "orientation_cmds" in obs_component:
                self.roll_pitch_yaw_cmd = torch.zeros(1, 3, device= self.model_device, dtype= torch.float32)
        
        # controls
        self.control_type = self.cfg["control"]["control_type"]
        if not (self.control_type == "P"):
            raise NotImplementedError("Only position control is supported for now.")
        self.p_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i] # set p_gains in simulation order
            for k, v in self.cfg["control"]["stiffness"].items():
                if k in name:
                    self.p_gains.append(v)# NOTE
                    break # only one match
        self.p_gains = torch.tensor(self.p_gains, device= self.model_device, dtype= torch.float32)
        self.d_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i] # set d_gains in simulation order
            for k, v in self.cfg["control"]["damping"].items():
                if k in name:
                    self.d_gains.append(v) #NOTE
                    break
        self.d_gains = torch.tensor(self.d_gains, device= self.model_device, dtype= torch.float32)
        self.default_dof_pos = torch.zeros(self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_pos_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_vel_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            default_joint_angle = self.cfg["init_state"]["default_joint_angles"][name]
            # in simulation order.
            self.default_dof_pos[i] = default_joint_angle
        self.computer_clip_torque = self.cfg["control"].get("computer_clip_torque", True)
        self.get_logger().info("Computer Clip Torque (onboard) is " + str(self.computer_clip_torque))
        self.torque_limits = getattr(RobotCfgs, self.robot_class_name).torque_limits.to(self.model_device)
        if self.computer_clip_torque:
            assert hasattr(self, "torque_limits") and (len(self.torque_limits) == self.NUM_DOF), f"torque_limits must be set with the length of {self.NUM_DOF} if computer_clip_torque is True"
            self.get_logger().info("[Env] torque limit: " + ",".join("{:.1f}".format(x) for x in self.torque_limits))
        
        # actions
        self.num_actions = self.NUM_ACTIONS
        self.action_scale = self.cfg["control"]["action_scale"]
        self.get_logger().info("[Env] action scale: {:.1f}".format(self.action_scale))
        self.clip_actions = self.cfg["normalization"]["clip_actions"]
        if self.cfg["normalization"].get("clip_actions_method", None) == "hard":
            self.get_logger().info("clip_actions_method with hard mode")
            self.get_logger().info("clip_actions_high: " + str(self.cfg["normalization"]["clip_actions_high"]))
            self.get_logger().info("clip_actions_low: " + str(self.cfg["normalization"]["clip_actions_low"]))
            self.clip_actions_method = "hard"
            self.clip_actions_low = torch.tensor(self.cfg["normalization"]["clip_actions_low"], device= self.model_device, dtype= torch.float32)
            self.clip_actions_high = torch.tensor(self.cfg["normalization"]["clip_actions_high"], device= self.model_device, dtype= torch.float32)
        else:
            self.get_logger().info("clip_actions_method is " + str(self.cfg["normalization"].get("clip_actions_method", None)))
        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)
            

        # hardware related, in simulation order
        self.joint_limits_high = getattr(RobotCfgs, self.robot_class_name).joint_limits_high.to(self.model_device)
        self.joint_limits_low = getattr(RobotCfgs, self.robot_class_name).joint_limits_low.to(self.model_device)
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.dof_pos_protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.dof_pos_protect_ratio

    def start_ros_handlers(self):
        """ after initializing the env and policy, register ros related callbacks and topics
        """

        # ROS publishers
        qos_profile = QoSProfile(
            reliability = ReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=2
        )
        self.low_cmd_pub = self.create_publisher(
            Action,
            self.low_cmd_topic,
            qos_profile
        )

        # ROS subscribers
        self.sub_action = self.create_subscription(
            State,
            self.low_state_topic,
            self._low_state_callback,
            1
        )
        self.joy_stick_sub = self.create_subscription(
            UserCommand,
            self.joy_stick_topic,
            self._joy_stick_callback,
            1
        )

        if self.forward_depth_topic is not None:
            self.forward_camera_sub = self.create_subscription(
                Image,
                self.forward_depth_topic,
                self._forward_depth_callback,
                1
            )

        if self.forward_depth_embedding_topic is not None and "forward_depth" in self.replace_obs_with_embeddings:
            self.forward_depth_embedding_sub = self.create_subscription(
                Float32MultiArray,
                self.forward_depth_embedding_topic,
                self._forward_depth_embedding_callback,
                1,
            )

        self.get_logger().info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.get_logger().warn(f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep safe.")
        else:
            self.get_logger().warn(f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be safe.")
        while rclpy.ok():
            rclpy.spin_once(self)
            if hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer"):
                break
        self.get_logger().info("Low state message received, the robot is ready to go.")


    """ ROS callbacks and handlers that update the buffer """
    def _low_state_callback(self, msg):
        """ store and handle proprioception data """
        self.low_state_buffer = msg # keep the latest low state

        # refresh dof_pos and dof_vel
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_pos_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].q * self.dof_signs[sim_idx]
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_vel_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].dq * self.dof_signs[sim_idx]
        # automatic safety check
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            if self.dof_pos_[0, sim_idx] > self.joint_pos_protect_high[sim_idx] or \
                self.dof_pos_[0, sim_idx] < self.joint_pos_protect_low[sim_idx]:
                self.get_logger().error(f"Joint {sim_idx} (sim), {real_idx} (real) position out of range at {self.low_state_buffer.motor_state[real_idx].q}")
                self.get_logger().error(f"Joint {sim_idx} (sim), {real_idx} (real) position out of range at {self.dof_pos_[0,sim_idx]}; high_protect at {self.joint_pos_protect_high[sim_idx]}, low_protect at {self.joint_pos_protect_low[sim_idx]}")
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()
                raise SystemExit()

    def _joy_stick_callback(self, msg):
        self.joy_stick_buffer = msg
        self.xyyaw_command = torch.tensor([msg.vx, msg.vy, msg.wz], device= self.model_device, dtype= torch.float32)
        if(msg.motion_mode==-1):
            self.get_logger().warn("Shutdown button was pressed, the motors and this process shuts down.")
            self._turn_off_motors()
            raise SystemExit()

        #self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))


    def _forward_depth_callback(self, msg):
        """ store and handle depth camera data """
        pass

    def _forward_depth_embedding_callback(self, msg):
        self.forward_depth_embedding_buffer = torch.tensor(msg.data, device= self.model_device, dtype= torch.float32).view(1, -1)

    """ Done: ROS callbacks and handlers that update the buffer """

    """ refresh observation buffer and corresponding sub-functions """
    def _get_lin_vel_obs(self):
        return torch.zeros(1, 3, device= self.model_device, dtype= torch.float32)
    
    def _get_ang_vel_obs(self):
        ang_vel_obs = torch.tensor([self.low_state_buffer.imu_state.gyroscope.x, 
                                    self.low_state_buffer.imu_state.gyroscope.y, 
                                    self.low_state_buffer.imu_state.gyroscope.z],
                                    device=self.model_device, 
                                    dtype=torch.float32).unsqueeze(0)
        return ang_vel_obs

    def _get_projected_gravity_obs(self):
        quat_xyzw = torch.tensor([
            self.low_state_buffer.imu_state.quaternion.x,
            self.low_state_buffer.imu_state.quaternion.y,
            self.low_state_buffer.imu_state.quaternion.z,
            self.low_state_buffer.imu_state.quaternion.w,
        ], device= self.model_device, dtype= torch.float32).unsqueeze(0)
        return quat_rotate_inverse(
            quat_xyzw,
            self.gravity_vec,
        )

    def _get_commands_obs(self):
        return self.xyyaw_command.unsqueeze(0) # (1, 3)

    def _get_dof_pos_obs(self):
        return self.dof_pos_ - self.default_dof_pos.unsqueeze(0)

    def _get_dof_vel_obs(self):
        return self.dof_vel_

    def _get_last_actions_obs(self):
        return self.actions.view(1,self.NUM_DOF)

    def _get_forward_depth_embedding_obs(self):
        return self.forward_depth_embedding_buffer
    
    def _get_forward_depth_obs(self):
        raise NotImplementedError()
    
    def _get_orientation_cmds_obs(self):
        return quat_rotate_inverse(
            quat_from_euler_xyz(self.roll_pitch_yaw_cmd[:, 0], self.roll_pitch_yaw_cmd[:, 1], self.roll_pitch_yaw_cmd[:, 2]),
            self.gravity_vec,
        )

    def _get_height_measurements_obs(self):
        #print("warning, height measurement is not valid on real robot")
        # NOTE, this is only for plane terrain
        return torch.zeros(1, len(self.cfg["terrain"]["measured_points_x"]), len(self.cfg["terrain"]["measured_points_y"])).view(1,-1)

    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = 0
        for k, v in obs_segments.items():
            num_obs += np.prod(v)
        return num_obs
        
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "lin_vel" in components:
            print("\033[40m Warning: lin_vel is not typically available or accurate enough on the real robot. Will return zeros.\033[0m")
            segments["lin_vel"] = (3,)
        if "ang_vel" in components:
            segments["ang_vel"] = (3,)
        if "projected_gravity" in components:
            segments["projected_gravity"] = (3,)
        if "commands" in components:
            segments["commands"] = (3,)
        if "dof_pos" in components:
            segments["dof_pos"] = (self.NUM_DOF,)
        if "dof_vel" in components:
            segments["dof_vel"] = (self.NUM_DOF,)
        if "last_actions" in components:
            segments["last_actions"] = (self.NUM_ACTIONS,)
        if "height_measurements" in components:
            print("\033[40m Warning: height_measurements is not typically available on the real robot.\033[0m")
            segments["height_measurements"] = (1, len(self.cfg["terrain"]["measured_points_x"]), len(self.cfg["terrain"]["measured_points_y"]))
        if "forward_depth" in components:
            if "output_resolution" in self.cfg["sensor"]["forward_camera"]:
                segments["forward_depth"] = (1, *self.cfg["sensor"]["forward_camera"]["output_resolution"])
            else:
                segments["forward_depth"] = (1, *self.cfg["sensor"]["forward_camera"]["resolution"])
        if "base_pose" in components:
            segments["base_pose"] = (6,) # xyz + rpy
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            print("Warning: height_measurements is not typically available on the real robot.")
            segments["robot_config"] = (1 + 3 + 1 + self.NUM_ACTIONS,)

        """ NOTE: The following components are not directly set in legged_robot.py.
        Please check the order or extend the class implementation if needed.
        """
        if "joints_target" in components:
            # o[0] for target value, 0[1] for wether the target should be tracked (1) or not (0)
            segments["joints_target"] = (2, self.NUM_DOF)
        if "projected_gravity_target" in components:
            # projected_gravity for which the robot should track the target
            # last value as a mask of whether to follow the target or not
            segments["projected_gravity_target"] = (3+1,)
        if "orientation_cmds" in components:
            segments["orientation_cmds"] = (3,)
        return segments
    
    def get_obs(self, obs_segments= None):
        """ Extract from the buffers and build the 1d observation tensor
        Each get ... obs function does not do the obs_scale multiplication.
        NOTE: obs_buffer has the batch dimension, whose size is 1.
        """
        if obs_segments is None:
            obs_segments = self.obs_segments
        obs_buffer = []
        for k, v in obs_segments.items():
            if k in self.replace_obs_with_embeddings:
                obs_component_value = getattr(self, "_get_" + k + "_embedding_obs")()
            else:
                obs_component_value = getattr(self, "_get_" + k + "_obs")() * self.obs_scales.get(k, 1.0)
            obs_buffer.append(obs_component_value)
        try:
            obs_buffer = torch.cat(obs_buffer, dim=1)
        except Exception as e:
            import pdb;pdb.set_trace()

        obs_buffer = torch.clamp(obs_buffer, -self.clip_obs, self.clip_obs)
        return obs_buffer
    """ Done: refresh observation buffer and corresponding sub-functions """

    """ Control related functions """
    def clip_action_before_scale(self, action):
        action = torch.clip(action, -self.clip_actions, self.clip_actions)
        if getattr(self, "clip_actions_method", None) == "hard":
            action = torch.clip(action, self.clip_actions_low, self.clip_actions_high)
        return action

    def clip_by_torque_limit(self, actions_scaled):
        """ Different from simulation, we reverse the process and clip the actions directly,
        so that the PD controller runs in robot but not our script.
        """
        control_type = self.cfg["control"]["control_type"]
        if control_type == "P":
            p_limits_low = (-self.torque_limits) + self.d_gains*self.dof_vel_
            p_limits_high = (self.torque_limits) + self.d_gains*self.dof_vel_
            actions_low = (p_limits_low/self.p_gains) - self.default_dof_pos + self.dof_pos_
            actions_high = (p_limits_high/self.p_gains) - self.default_dof_pos + self.dof_pos_
        else:
            raise NotImplementedError

        return torch.clip(actions_scaled, actions_low, actions_high)

    def send_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        self.actions = self.clip_action_before_scale(actions)
        if self.computer_clip_torque:
            clipped_scaled_action = self.clip_by_torque_limit(actions * self.action_scale)
        else:
            self.get_logger().warn("Computer Clip Torque is False, the robot may be damaged.", throttle_duration_sec= 1)
            clipped_scaled_action = actions * self.action_scale
        robot_coordinates_action = clipped_scaled_action + self.default_dof_pos.unsqueeze(0)

        self._publish_legs_cmd(robot_coordinates_action[0])

    """ Done: Control related functions """

    """ functions that actually publish the commands and take effect """
    def _publish_legs_cmd(self, robot_coordinates_action: torch.Tensor):
        """ Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_DOF,), in simulation order.
        """
        for sim_idx in range(self.NUM_DOF):
            tmp = MotorAction()
            real_idx = self.dof_map[sim_idx]
            if not self.dryrun:
                tmp.mode = self.turn_on_motor_mode[sim_idx]
            tmp.id = sim_idx+1
            tmp.q = robot_coordinates_action[sim_idx].item() * self.dof_signs[sim_idx]
            tmp.dq = 0.
            tmp.tau = 0.
            tmp.kp = self.p_gains[sim_idx].item()
            tmp.kd = self.d_gains[sim_idx].item()
            self.low_cmd_buffer.motor_action[real_idx]=tmp
        
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """ Turn off the motors """
        for sim_idx in range(self.NUM_DOF):
            tmp = MotorAction()
            real_idx = self.dof_map[sim_idx]
            tmp.mode = 0x00
            tmp.id = sim_idx+1
            tmp.q = 0.
            tmp.dq = 0.
            tmp.tau = 0.
            tmp.kp = 0.
            tmp.kd = 0.
            self.low_cmd_buffer.motor_action[real_idx]=tmp
        self.low_cmd_pub.publish(self.low_cmd_buffer)
    """ Done: functions that actually publish the commands and take effect """

