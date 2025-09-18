"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv



def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int) -> None:
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        return weight
    else:
        return env.reward_manager.get_term_cfg(term_name).weight






def update_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    penalty_level_down_threshold: float,
    penalty_level_up_threshold: float,
    penalty_level_degree: float,
    min_penalty_scale: float,
    max_penalty_scale: float,
    term_names: list
):
    """
    Update average episode length and in turn penalty curriculum.

    This function is rewritten from update_average_episode_length of legged_gym.

    When the policy is not able to track the motions, we reduce the penalty to help it explore more actions. When the
    policy is able to track the motions, we increase the penalty to smooth the actions and reduce the maximum action
    it uses.
    """
    N = env.scene.num_envs if env_ids is None else len(env_ids)
    if not hasattr(env, "penalty_scale"):
        env.penalty_scale = 1.0

    env.average_episode_length = torch.mean(env.episode_length_buf[env_ids], dtype=torch.float)

    if env.average_episode_length < penalty_level_down_threshold:
        env.penalty_scale *= 1 - penalty_level_degree
    elif env.average_episode_length > penalty_level_up_threshold:
        env.penalty_scale *= 1 + penalty_level_degree
    env.penalty_scale = np.clip(env.penalty_scale, min_penalty_scale, max_penalty_scale)

    for term_name in term_names:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight *= env.penalty_scale
        env.reward_manager.set_term_cfg(term_name, term_cfg)




# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

def command_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str, max_curriculum: float = 1.0
) -> None:
    """Curriculum based on the tracking reward of the robot when commanded to move at a desired velocity.

    This term is used to increase the range of commands when the robot's tracking reward is above 80% of the
    maximum.

    Returns:
        The cumulative increase in velocity command range.
    """
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    delta_range = torch.tensor([-0.1, 0.1], device=env.device)
    if not hasattr(env, "delta_lin_vel"):
        env.delta_lin_vel = torch.tensor(0.0, device=env.device)
    # If the tracking reward is above 80% of the maximum, increase the range of commands
    if torch.mean(episode_sums[env_ids]) / env.max_episode_length > 0.8 * reward_term_cfg.weight:
        lin_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        lin_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        base_velocity_ranges.lin_vel_x = torch.clamp(lin_vel_x + delta_range, -max_curriculum, max_curriculum).tolist()
        base_velocity_ranges.lin_vel_y = torch.clamp(lin_vel_y + delta_range, -max_curriculum, max_curriculum).tolist()
        env.delta_lin_vel = torch.clamp(env.delta_lin_vel + delta_range[1], 0.0, max_curriculum)
    return env.delta_lin_vel





def adaptive_command_range_scaling(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    command_name: str,
    reward_thresholds: tuple[float, float],
    scale_limits: tuple[float, float],
    ):
    """
    Adaptively scale command range based on average reward.

    Args:
        env: The learning environment.
        env_ids: Not used (affects all environments).
        command_name: Name of the command term, e.g., 'base_velocity'.
        reward_thresholds: (low, high) thresholds to map reward to scale.
                           Below `low` → scale_min, above `high` → scale_max.
        scale_limits: (scale_min, scale_max), e.g., (0.2, 1.0)
    """
    # Compute average reward across environments
    term_value = env.reward_manager._step_reward[:, env.reward_manager._term_names.index(term_name)].clone().detach().mean().item()
    print(f"term_name:{term_value}")
    lin_vel_x = env.command_manager.get_term_cfg(command_name).cfg.ranges.lin_vel_x
    print(f"command_cfg:{lin_vel_x}")
    return


    # Linearly map reward to scale
    r_min, r_max = reward_thresholds
    s_min, s_max = scale_limits
    scale = (avg_reward - r_min) / (r_max - r_min)
    scale = max(0.0, min(1.0, scale))  # clip to [0, 1]
    scale = s_min + scale * (s_max - s_min)

    # Update command range
    for key in ["lin_x", "lin_y", "ang_z"]:
        if hasattr(command_cfg.ranges, key):
            orig_range = getattr(command_cfg.ranges, key)
            center = 0.5 * (orig_range[0] + orig_range[1])
            half_span = 0.5 * (orig_range[1] - orig_range[0]) * scale
            scaled_range = [center - half_span, center + half_span]
            setattr(command_cfg.ranges, key, scaled_range)

    env.command_manager.set_command_cfg(command_name, command_cfg)


