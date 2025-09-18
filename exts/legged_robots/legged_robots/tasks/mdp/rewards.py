from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, euler_xyz_from_quat, quat_mul, convert_quat, axis_angle_from_quat, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


"""
Tao Sun add the following rewards

"""

def reward_action_smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    if not hasattr(env, "prev_prev_action"):
        env.prev_prev_action = torch.zeros_like(env.action_manager.prev_action)

    diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.prev_prev_action)
    diff = diff * (env.action_manager.prev_action != 0)  # ignore first step
    diff = diff * (env.prev_prev_action!= 0)  # ignore second step
    
    env.prev_prev_action[:] = env.action_manager.prev_action

    return torch.sum(diff, dim=1)

def energy_cost(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def track_style_goal_commands_exp(
        env: ManagerBasedRLEnv, std: float, command_name: str, fields: list, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    #asset.data.root_lin_vel_b[:, :2]
    current_state = asset.data.root_quat_w

    if hasattr(asset.data,"style_goal"):
        goal_states = asset.data.style_goal
    else:
        goal_states = torch.zeros_like(current_state)

    # compute the error
    track_error = torch.sum(
            torch.square(goal_states - current_state),
            dim=1,
    )
    return torch.exp(-track_error / std**2)


def track_fields_exp(
    env: ManagerBasedRLEnv,
    std: float,
    track_type: str,
    field_name: list = None,
    min_std: float = 0.05,
    max_std: float = 20,
    tracking_term: str = "joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of expressive fields (joint/link pos/vel) using exponential kernel."""
    # Validate std
    if std <= 0:
        raise ValueError("std must be positive")

    # Extract asset and initialize variables
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(env, "ref_motion"):
        raise ValueError(f"{type(env)} has no ref_motion")

    #I) joint pos and velocity
    if track_type == "joint_pos":
        try:
            field_index = [asset.data.joint_names.index(key) for key in field_name]
            expre_field_index = [env.ref_motion.trajectory_fields.index(key+"_dof_pos") for key in field_name]
        except ValueError as e:
            raise ValueError(f"Invalid joint name in field_name: {e}")
        current_states = asset.data.joint_pos[:, field_index]
        goal_states = env.ref_motion.data[:, expre_field_index]
    elif track_type == "joint_vel":
        try:
            field_index = [asset.data.joint_names.index(key) for key in field_name]
            expre_field_index = [env.ref_motion.trajectory_fields.index(key+"_dof_vel") for key in field_name]
        except ValueError as e:
            raise ValueError(f"Invalid joint name in field_name: {e}")
        current_states = asset.data.joint_vel[:, field_index]
        goal_states = env.ref_motion.data[:, expre_field_index]
    #II) link pos and velocity
    elif track_type == "link_pos_w":
        try:
            field_index = [asset.body_names.index(name) for name in field_name]
            expre_field_index = [env.ref_motion.trajectory_fields.index(key1+key2) for key1 in field_name for key2 in ["_pos_x_w", "_pos_y_w", "_pos_z_w"]]
        except ValueError as e:
            raise ValueError(f"Invalid body name in field_name: {e}")
        body_pos_w = asset.data.body_pos_w[:, field_index, :] - env.scene.env_origins.unsqueeze(1)
        current_states = body_pos_w.reshape(-1, len(field_index) * 3)
        goal_states = env.ref_motion.data[:, expre_field_index].reshape(-1, len(field_index)*3)
    elif track_type == "link_pos_b":
        try:
            field_index = [asset.body_names.index(name) for name in field_name]
            expre_field_index = [env.ref_motion.trajectory_fields.index(key1+key2) for key1 in field_name for key2 in ["_pos_x_b", "_pos_y_b", "_pos_z_b"]]
        except ValueError as e:
            raise ValueError(f"Invalid body name in field_name: {e}")
        body_pos_w = asset.data.body_pos_w[:, field_index, :] - env.scene.env_origins.unsqueeze(1)
        root_pos_world = asset.data.root_pos_w  # (envs, 3)
        root_quat_world = asset.data.root_quat_w # (envs, 4)
        body_pos_w = body_pos_w - root_pos_world.unsqueeze(1)         # (envs, bodies, 3)
        root_quat_world_expanded = root_quat_world.unsqueeze(1).expand(-1, body_pos_w.shape[1], -1).reshape(-1, 4) # (envs_bodies,4)
        body_pos_b = quat_rotate_inverse(root_quat_world_expanded, body_pos_w.reshape(-1,3))  # (envs*bodies, 3)
        current_states = body_pos_b.reshape(-1, len(field_index) * 3)
        goal_states = env.ref_motion.data[:,expre_field_index].reshape(-1, len(field_index)*3)
    elif track_type == "link_vel_w":
        try:
            field_index = [asset.body_names.index(name) for name in field_name]
            expre_field_index = [env.ref_motion.trajectory_fields.index(key1+key2) for key1 in field_name for key2 in ["_vel_x_w", "_vel_y_w", "_vel_z_w"]]
        except ValueError as e:
            raise ValueError(f"Invalid body name in field_name: {e}")
        body_lin_vel_w = asset.data.body_lin_vel_w[:, field_index, :]
        current_states = body_lin_vel_w.reshape(-1, len(field_index) * 3)
        goal_states = env.ref_motion.data[:, expre_field_index].reshape(-1, len(field_index)*3)
    elif track_type == "link_vel_b":
        try:
            field_index = [asset.body_names.index(name) for name in field_name]
            expre_field_index = [env.ref_motion.trajectory_fields.index(key1+key2) for key1 in field_name for key2 in ["_vel_x_b", "_vel_y_b", "_vel_z_b"]]
        except ValueError as e:
            raise ValueError(f"Invalid body name in field_name: {e}")
        body_lin_vel_w = asset.data.body_lin_vel_w[:, field_index, :]
        root_quat_world = asset.data.root_quat_w # (envs, 4)
        root_quat_world_expanded = root_quat_world.unsqueeze(1).expand(-1, body_lin_vel_w.shape[1], -1).reshape(-1, 4) # (envs_bodies,4)
        body_lin_vel_b = quat_rotate_inverse(root_quat_world_expanded, body_lin_vel_w.reshape(-1,3)) # (env*bodies,3)
        current_states = body_lin_vel_b.reshape(-1, len(field_index) * 3)
        goal_states = env.ref_motion.data[:, expre_field_index].reshape(-1, len(field_index)*3)
    #III) root pos and velocity
    elif track_type == "root_pos_w":
        current_states = asset.data.root_pos_w - env.scene.env_origins
        goal_states = env.ref_motion.root_pos_w 
    elif track_type == "root_quat_w":
        current_states = asset.data.root_quat_w
        goal_states = env.ref_motion.root_quat_w 
    elif track_type == "root_lin_vel_w":
        current_states = asset.data.root_lin_vel_w
        goal_states = env.ref_motion.root_lin_vel_w 
    elif track_type == "root_lin_vel_b":
        current_states = asset.data.root_lin_vel_b
        goal_states = env.ref_motion.root_lin_vel_b
    elif track_type == "root_ang_vel_w":
        current_states = asset.data.root_ang_vel_w
        goal_states = env.ref_motion.root_ang_vel_w 
    elif track_type == "root_ang_vel_b":
        current_states = asset.data.root_ang_vel_b
        goal_states = env.ref_motion.root_ang_vel_b
    elif track_type == "root_rotation_w":
        current_states = asset.data.root_quat_w
        goal_states = env.ref_motion.root_quat_w
        diff_global_body_rot = quat_mul(goal_states, quat_conjugate(current_states))
        diff_axis_angle = axis_angle_from_quat(diff_global_body_rot)
        diff_angle = torch.norm(diff_axis_angle, dim=-1).unsqueeze(1)  # shape (num_env,1)
        current_states = diff_angle
        goal_states = torch.zeros_like(current_states)
    else:
        raise ValueError(f"Unknown type: {type}")

    # Compute error and reward
    assert current_states.shape == goal_states.shape, f"Shape mismatch: {current_states.shape} vs {goal_states.shape}"

    #error = torch.sum((goal_states - current_states)**2, dim=1)
    #return torch.exp(-error / (std**2 + 1e-8))  # Add epsilon for stability
    if not hasattr(env,"sigma_tracker"):
        env.sigma_tracker = AdaptiveSigmaTracker(device=env.device,sigma_min=min_std, sigma_max=max_std)

    error = (goal_states - current_states)**2

    sigma = env.sigma_tracker.update(tracking_term, error, init_sigma=std)
    reward = torch.exp(-torch.mean(error,dim=-1)/ (sigma + 1e-6))
    return reward  # shape: [num_envs]






def feet_stumble(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold_ratio: float = 0.6
) -> torch.Tensor:
    """
    Penalize stumbling by detecting if contact forces in the XY plane greatly exceed vertical (Z) contact forces.

    Args:
        env (ManagerBasedRLEnv): The environment.
        asset_cfg (SceneEntityCfg): Scene entity (e.g., robot) with defined body_ids and name.
        threshold_ratio (float): Ratio threshold to consider a contact as a stumble.

    Returns:
        torch.Tensor: A float tensor of shape (num_envs,) with 1.0 if stumble detected, else 0.0.
        """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # [B, BODIES, 2] -> [B, BODIES, 1]
    contact_forces_xy = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2].norm(dim=-1)
    contact_forces_z  = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    # Compare norm of xy vs z component per body, then reduce across time+body (dim=1)
    is_stumble = torch.any(contact_forces_xy > threshold_ratio * contact_forces_z, dim=(1))
    return is_stumble.float()





def feet_parallel_v1(  # 奖励脚部平行度
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name] # 获取机器人资产对象
    body_quat1 = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].reshape(-1, 4) # feet orientation
    body_quat2 = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :].reshape(-1, 4)
    asset_roll1, asset_pitch1, asset_yaw1 = euler_xyz_from_quat(body_quat1)
    asset_roll2, asset_pitch2, asset_yaw2 = euler_xyz_from_quat(body_quat2)
    roll_pitch1 = torch.abs(torch.stack([asset_roll1, asset_pitch1], dim=-1)) # 取roll和pitch的绝对值
    roll_pitch2 = torch.abs(torch.stack([asset_roll2, asset_pitch2], dim=-1)) # 取roll和pitch的绝对值
    reward = torch.exp(-(torch.sum(torch.square(roll_pitch1), dim=1) + torch.sum(torch.square(roll_pitch2), dim=1))/(std**2))# 计算脚部平行度奖励（绝对值平方和）
    #reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




class AdaptiveSigmaTracker:
    def __init__(self, alpha=0.95, sigma_min=0.05, sigma_max=30.0, scale=1.5, device="cuda:0"):
        """
        用于根据误差动态调整 sigma，用于 reward 计算稳定性。
        参数:
            alpha (float): 滑动平均的平滑系数，越接近 1 越稳定
            sigma_min (float): 最小 sigma
            sigma_max (float): 最大 sigma
            scale (float): 将误差放大以更新 sigma
            device (str): CUDA 或 CPU
        """
        self.alpha = alpha
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.scale = scale
        self.device = device
        self.tracking_sigma = {}  # 保存每个 term 的 sigma
        self.tracking_mean_error = {}
        self.episodic_sum_tracking_error = {}

    def update(self, term: str, error: torch.Tensor, init_sigma: float =0.25) -> torch.Tensor:
        """
        根据误差更新指定 tracking term 的 sigma。

        参数:
            term (str): tracking term 名称
            error (Tensor): shape [num_envs, dim] 的误差张量
            返回:
                sigma (Tensor): 更新后的 sigma, shape [dim]
        """
        with torch.no_grad():
            mean_error = torch.mean(error, dim=-1)  # shape [env_num,]
            scaled_error = mean_error * self.scale

            if term not in self.episodic_sum_tracking_error:
                self.episodic_sum_tracking_error[term] = mean_error
            else:
                self.episodic_sum_tracking_error[term] += mean_error

            if term not in self.tracking_sigma:
                self.tracking_sigma[term] = init_sigma*torch.ones_like(scaled_error) 
            else:
                prev_sigma = self.tracking_sigma[term]
                new_sigma = self.alpha * prev_sigma + (1 - self.alpha) * scaled_error
                self.tracking_sigma[term] = torch.clamp(
                    new_sigma,
                    min=self.sigma_min,
                    max=self.sigma_max
                )
            self.tracking_mean_error[term] = mean_error
            return self.tracking_sigma[term] # shaep [num_env]


    def get(self, term: str) -> torch.Tensor:
        return self.tracking_sigma.get(term,self.sigma_max)
