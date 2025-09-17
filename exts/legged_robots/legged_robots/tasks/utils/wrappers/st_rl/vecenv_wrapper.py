# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRlEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


import gymnasium as gym
import torch

from st_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from collections import deque

#from legged_robots.data_manager.motion_loader import RefMotionLoader
from refmotion_manager.motion_loader import RefMotionLoader

from collections import deque

import time
from isaaclab.managers import SceneEntityCfg

class StRlVecEnvWrapper(VecEnv):
    """Wrapper class to add metrics and logging capabilities to an RL environment."""

    def __init__(self, env : RslRlVecEnvWrapper):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        if not isinstance(env, RslRlVecEnvWrapper):
            raise ValueError(
                "The environment must be inherited from RslRlVecEnvWrapper. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.device = self.env.device
        self.max_episode_length = self.env.max_episode_length
        print(f"\033[93m Max episode length: {self.max_episode_length}\033[0m")
        delay_steps = 1
        print(f"\033[93m obs delay time: {(delay_steps-1)*self.cfg.sim.dt*self.cfg.decimation}\033[0m")
        self.obs_buf = deque(maxlen=delay_steps)  # delay_steps = int(delay_time / dt)

        if hasattr(self.cfg, "ref_motion"):
            self.unwrapped.ref_motion = RefMotionLoader(self.cfg.ref_motion)
            self.unwrapped.ref_motion.step()
            self.unwrapped.ref_motion.reset()



    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        obs, extras = self.env.get_observations()

        if hasattr(self.unwrapped, "ref_motion"):
            extras["ref_motion"] = self.unwrapped.ref_motion.amp_expert

        return obs, extras


    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:

        if hasattr(self.unwrapped, "ref_motion"):
            # base velocity should come from demo data if using imiattion learning
            self.unwrapped.scene["robot"].data.demo_base_velocity_w = self.unwrapped.ref_motion.base_velocity_w
            self.unwrapped.scene["robot"].data.demo_base_velocity_b = self.unwrapped.ref_motion.base_velocity_b

            # for env initialization
            if self.cfg.ref_motion.init_state_fields is not None:
                self.unwrapped.scene["robot"].data.demo_init_states = self.unwrapped.ref_motion.init_states

            # for amp goal/commands
            if self.cfg.ref_motion.style_goal_fields is not None:
                self.unwrapped.scene["robot"].data.style_goal = self.unwrapped.ref_motion.style_goal

            # for expressive goal/commands
            if self.cfg.ref_motion.expressive_goal_fields is not None:
                self.unwrapped.scene["robot"].data.expressive_goal = self.unwrapped.ref_motion.expressive_goal

            if self.cfg.using_ref_motion_in_actions:
                joint_names = self.unwrapped.scene["robot"].data.joint_names
                expre_field_index = [self.unwrapped.ref_motion.trajectory_fields.index(key+"_dof_pos") for key in joint_names]
                ref_actions = self.unwrapped.ref_motion.data[:, expre_field_index]
                actions = 0.5*actions + ref_actions
            
        obs, rew, dones, extras = self.env.step(actions)

        # move base_velocity to the extras dict
        extras["log"]["Episode_Command/lin_vel_x_max"] = self.unwrapped.command_manager.get_command("base_velocity").max(dim=0).values.clone().detach()[0]
        extras["log"]["Episode_Command/lin_vel_y_max"] = self.unwrapped.command_manager.get_command("base_velocity").max(dim=0).values.clone().detach()[1]
        extras["log"]["Episode_Command/lin_vel_z_max"] = self.unwrapped.command_manager.get_command("base_velocity").max(dim=0).values.clone().detach()[2]
        if hasattr(self.unwrapped, "sigma_tracker"):
            for key, value in self.unwrapped.sigma_tracker.tracking_sigma.items():
                extras["log"]["Episode_Tracking_Sigma/tracking_sigma_"+str(key)] = value.mean()

            for key, value in self.unwrapped.sigma_tracker.tracking_mean_error.items():
                extras["log"]["Episode_Tracking_Error/tracking_mean_error_"+str(key)] = value.mean()

            for key in self.unwrapped.sigma_tracker.episodic_sum_tracking_error:
                self.unwrapped.sigma_tracker.episodic_sum_tracking_error[key][dones] = 0.0
        
        if hasattr(self.cfg, "ref_motion"):
            self.unwrapped.ref_motion.step()
            reset_mask = (self.unwrapped.ref_motion.frame_idx >= self.unwrapped.ref_motion.augment_frame_num - 1)
            reset_mask = torch.logical_or(reset_mask, dones)
            self.unwrapped.ref_motion.reset(reset_mask)
            extras["ref_motion"] = self.unwrapped.ref_motion.amp_expert

        if hasattr(self.unwrapped, "penalty_scale"):
            extras["log"]["Episode_Penalty_Scale"] = self.unwrapped.penalty_scale

        # Append the latest observation
        self.obs_buf.append(obs.clone())

        return self.obs_buf[0], rew, dones, extras

    """
    directly reset
    """
    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        if hasattr(self.unwrapped, "ref_motion"):
            self.unwrapped.ref_motion.reset()
        return self.env.reset()


    """
    Properties
    """
    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped


    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    @property
    def num_obs(self) -> int:
        return self.env.num_obs

    @property
    def num_privileged_obs(self) -> int:
        return self.env.num_privileged_obs

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def close(self):  # noqa: D102
        return self.env.close()

    @property
    def CoT(self) -> torch.Tensor:
        """Calculate the Cost of Transport (CoT)."""
        return torch.zeros(1)


    @property              
    def tracking_error(self) -> torch.Tensor:
        """Return the normalized tracking error (e.g. joint-level)."""
        if not hasattr(self.unwrapped, "sigma_tracker"):
            return torch.tensor(0.0, device=self.unwrapped.device)

        value = 0.0
        episode_lengths = torch.clamp(self.unwrapped.episode_length_buf, min=1).float()

        for key in ["upper_dof_pos", "lower_dof_pos"]:
            if key in self.unwrapped.sigma_tracker.episodic_sum_tracking_error:
                error_sum = self.unwrapped.sigma_tracker.episodic_sum_tracking_error[key]
                value += torch.mean(error_sum/episode_lengths)
        
        return torch.tensor(value, device=self.unwrapped.device)



# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRlEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""

import torch
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.utils.math import euler_xyz_from_quat as get_euler_xyz

from legged_robots.tasks.utils.logger import Logger

class StRlMetricEnvWrapper(VecEnv):
    """Wrapper class to add metrics and logging capabilities to an RL environment."""

    def __init__(self, env: StRlVecEnvWrapper, enable_logger=False,log_dir=None):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            enable_logger: Whether to enable logging.
        """

        if not isinstance(env, StRlVecEnvWrapper):
            raise ValueError(
                "The environment must be inherited from StRlVecEnvWrapper. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.__dict__.update(env.__dict__)
        self.env = env
        self.device = self.env.device
        self.max_episode_length = self.env.max_episode_length

        asset_cfg = SceneEntityCfg("robot")
        asset = self.unwrapped.scene[asset_cfg.name]
        self.data = asset.data
        if "feet_contact_forces" in self.unwrapped.scene.keys():
            self.contact_sensor: ContactSensor = self.unwrapped.scene["feet_contact_forces"]

        self.joint_names = self.unwrapped.scene["robot"].joint_names
        self.default_body_mass = torch.sum(self.data.default_mass, dim=1).to(self.device)
        self.payloads = torch.zeros_like(self.default_body_mass)
        self.g = 9.8  # Gravitational constant
        self.start_log = False
        self.finish_log = False

        if enable_logger:
            self.logger = Logger(self.unwrapped.sim.cfg.dt, log_dir=log_dir)
        else:
            self.logger = None

    @property
    def base_lin_vel(self) -> torch.Tensor:
        """Get the base linear velocity in the robot's frame."""
        vel_yaw = quat_rotate_inverse(yaw_quat(self.data.root_link_quat_w), self.data.root_com_lin_vel_w[:, :3])
        return vel_yaw
    
    @property
    def base_ang_vel(self) -> torch.Tensor:
        """Get the base angular velocity in the robot's frame."""
        ang_vel = quat_rotate_inverse(yaw_quat(self.data.root_link_quat_w), self.data.root_com_ang_vel_w[:, :3])
        return ang_vel


    @property
    def projected_gravity(self) -> torch.Tensor:
        """Get the gravity vector projected in the robot's base frame."""
        return self.data.projected_gravity_b


    @property
    def commands(self) -> torch.Tensor:
        """Get the commanded velocities."""
        command_name = "base_velocity"
        return self.unwrapped.command_manager.get_command(command_name)[:, :]

    @property
    def dof_pos(self) -> torch.Tensor:
        """Get the joint positions."""
        return self.data.joint_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        """Get the joint velocities."""
        return self.data.joint_vel

    @property
    def dof_torque(self) -> torch.Tensor:
        """Get the applied joint torques."""
        return self.data.applied_torque

    @property
    def power_consumption(self) -> torch.Tensor:
        """Calculate the power consumption of the robot."""
        return torch.sum(torch.abs(self.dof_torque) * torch.abs(self.dof_vel), dim=1)

    @property
    def CoT(self) -> torch.Tensor:
        """Calculate the Cost of Transport (CoT)."""
        P = self.power_consumption
        m = self.default_body_mass + self.payloads
        v = torch.norm(self.base_lin_vel[:, 0:2], dim=1)
        v = torch.where(v == 0, torch.ones_like(v), v)  # Avoid division by zero
        return P / (m * self.g * v)

    @property
    def lin_vel_rmsd(self) -> torch.Tensor:
        """Calculate the RMSD of linear velocity."""
        return ((self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2).cpu() ** 0.5

    def start_logging(self):
        print(f"[Logging] Start logging data")
        self.start_log=True
        self.finish_log=False


    def finish_logging(self):
        print(f"[Logging] Finish logging data")
        self.finish_log=True


    def log_update(self, robot_index=0):
        """Log the states of the robot.

        Args:
            robot_index: The index of the robot to log.
        """

        if self.logger is None:
            return

        states = {
            'command_x': self.commands[robot_index, 0].item(),
            'command_y': self.commands[robot_index, 1].item(),
            'command_yaw': self.commands[robot_index, 2].item(),
            'base_lin_vel_x': self.base_lin_vel[robot_index, 0].item(),
            'base_lin_vel_y': self.base_lin_vel[robot_index, 1].item(),
            'base_lin_vel_z': self.base_lin_vel[robot_index, 2].item(),
            'base_ang_vel_x': self.base_ang_vel[robot_index, 0].item(),
            'base_ang_vel_y': self.base_ang_vel[robot_index, 1].item(),
            'base_ang_vel_z': self.base_ang_vel[robot_index, 2].item(),
            'base_pro_gravity_x': self.projected_gravity[robot_index, 0].item(),
            'base_pro_gravity_y': self.projected_gravity[robot_index, 1].item(),
            'base_pro_gravity_z': self.projected_gravity[robot_index, 2].item(),
            'base_roll': get_euler_xyz(self.data.root_link_quat_w)[0][robot_index].item(),
            'base_pitch': get_euler_xyz(self.data.root_link_quat_w)[1][robot_index].item(),
            'base_yaw': get_euler_xyz(self.data.root_link_quat_w)[2][robot_index].item(),
        }

        # Log joint-specific states
        for idx in range(self.num_actions):
            states[f"{self.joint_names[idx]}_action"] = self.unwrapped.action_manager._action[robot_index,idx].item()
            states[f'{self.joint_names[idx]}_pos_target'] = self.data.joint_pos_target[robot_index, idx].item()
            states[f'{self.joint_names[idx]}_pos'] = self.dof_pos[robot_index, idx].item()
            states[f'{self.joint_names[idx]}_vel'] = self.dof_vel[robot_index, idx].item()
            states[f'{self.joint_names[idx]}_tau'] = self.dof_torque[robot_index, idx].item()

        
        # Log contact forces
        grf = self.contact_sensor.data.net_forces_w_history.mean(dim=1)
        for idx in range(grf.shape[1]):
            states[f'grf_x_{idx}'] = grf[:,idx,0].item()
            states[f'grf_y_{idx}'] = grf[:,idx,1].item()
            states[f'grf_z_{idx}'] = grf[:,idx,2].item()

        self.logger.update_states(states)

    def log_save(self):
        """Save the logger."""
        if self.logger is not None:
            self.logger.save()
            self.logger.reset()
            self.start_log = False
            
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        obs, extras = self.env.get_observations()
        return obs, extras


    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        return self.env.step(actions)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        return self.env.reset()


    def close(self):  # noqa: D102
        return self.env.close()


    """
    Properties
    """
    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped


    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg


    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    @property
    def num_obs(self) -> int:
        return self.env.num_obs

    @property
    def num_privileged_obs(self) -> int:
        return self.env.num_privileged_obs

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value


