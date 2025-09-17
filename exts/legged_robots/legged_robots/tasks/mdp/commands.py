# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import math
from dataclasses import MISSING
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

@configclass
class ExpressiveCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity command generator."""
    num_commands:int = MISSING


class BaseVelocityCommand(UniformVelocityCommand):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """
    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)


    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        if hasattr(self.robot.data,"demo_base_velocity_b"):
            self.vel_command[:, 0] = self.vel_command_b[:,0] + self.robot.data.demo_base_velocity_b[:,0]
            self.vel_command[:, 1] = self.vel_command_b[:,1] + self.robot.data.demo_base_velocity_b[:,1]
            self.vel_command[:, 2] = self.vel_command_b[:,2] + self.robot.data.demo_base_velocity_b[:,2]
        else:
            self.vel_command[:, 0] = self.vel_command_b[:, 0]
            self.vel_command[:, 1] = self.vel_command_b[:, 1]
            self.vel_command[:, 2] = self.vel_command_b[:, 2]

        return self.vel_command



class StyleCommand(UniformVelocityCommand):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """
    cfg: ExpressiveCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.style_commands = torch.zeros(self.num_envs, cfg.num_commands, device=self.device)

    def _update_command(self):
        # sample velocity commands
        # expressive commands
        if hasattr(self.robot.data,"style_goal"):
            self.style_commands = self.robot.data.style_goal
        else:
            warnings.warn("No style commands found in robot.data")

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        if hasattr(self.robot.data,"style_goal"):
            self.style_commands = self.robot.data.style_goal
        return self.style_commands


    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        #self.metrics["error_style_cmd"] += (
        #    torch.norm(self.style_commands[:, :] - self.robot.data.root_quat_w[:, :], dim=-1) / max_command_step
        #)

import warnings
class ExpressiveCommand(UniformVelocityCommand):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """
    cfg: ExpressiveCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: ExpressiveCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.expressive_commands = torch.zeros(self.num_envs, cfg.num_commands, device=self.device)

    def _update_command(self):
        # sample velocity commands
        # expressive commands
        if hasattr(self.robot.data,"expressive_goal"):
            self.expressive_commands = self.robot.data.expressive_goal
        else:
            warnings.warn("No expressive commands found in robot.data")

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        if hasattr(self.robot.data,"expressive_goal"):
            self.expressive_commands = self.robot.data.expressive_goal
        else:
            warnings.warn("No expressive commands found in robot.data")
        return self.expressive_commands


    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
