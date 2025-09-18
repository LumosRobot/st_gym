from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def tracking_error_adaptive_termination(
    env: ManagerBasedRLEnv,
    error_field: str = "lower_dof_pos",
    min_threshold: float = 0.2,
    max_threshold: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Terminate the episode if tracking error is too high with an adaptive threshold.
    The threshold decreases linearly over the episode time to enforce tighter tracking.
    """
    # make sure tracking error stats are available
    if not hasattr(env, "sigma_tracker"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env.sigma_tracker, "tracking_mean_error"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # retrieve current max error for each env
    mean_error = env.sigma_tracker.tracking_mean_error.get(error_field, None)
    if mean_error is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # compute adaptive threshold: linearly decay from max to min
    decay_ratio = env.episode_length_buf.float() / float(env.max_episode_length)
    decay_ratio = torch.clamp(decay_ratio, 0.0, 1.0)

    threshold = max_threshold - decay_ratio * (max_threshold - min_threshold)
    threshold = torch.clamp(threshold, min=min_threshold)

    # termination condition
    return mean_error > threshold

