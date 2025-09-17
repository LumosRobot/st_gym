from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_root_state_demo_traj(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    if hasattr(env, "ref_motion"):
        env.ref_motion.reset(env_ids)
        init_states = env.ref_motion.init_states[env_ids].clone()
        root_states = init_states[:,:13]
        positions = root_states[:,0:3] + env.scene.env_origins[env_ids]
        orientations = root_states[:,3:7]
        velocities = root_states[:,7:13]
    else:
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()
        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)




def reset_joints_by_demo_traj(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_num = len(asset.data.joint_names)

    # get default joint state
    if hasattr(env, "ref_motion"):
        init_states = env.ref_motion.init_states[env_ids].clone()
        joint_pos = init_states[:,13:13+joint_num]
        joint_vel = init_states[:,13+joint_num:13+2*joint_num]
    else:
        # get default joint state
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_vel = asset.data.default_joint_vel[env_ids].clone()
        # scale these values randomly
        joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
        joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms()

    if not hasattr(env, "default_coms"):
        # Randomize robot base com
        env.default_coms = coms.clone()
        env.base_com_bias = torch.zeros((env.num_envs, 3), dtype=torch.float, device=coms.device)

    # apply randomization on default values
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (distribution_params[0].to(coms.device), distribution_params[1].to(coms.device))
    
    for idx in range(3):
        env.base_com_bias[env_ids, idx] = dist_fn(*(distribution_params[0][idx],distribution_params[1][idx]), (env_ids.shape[0], 1), device=coms.device).squeeze()

    # sample from the given range
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    asset.root_physx_view.set_coms(coms, env_ids)

def randomize_action_noise_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the sample range of the added action noise by adding, scaling or setting random values.

    This function allows randomizing the scale of the sample range of the added action noise. The function
    samples random values from the given distribution parameters and adds, scales or sets the values into the
    simulation based on the operation.

    """

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    rfi_lim = env.default_rfi_lim.clone()

    dist_fn = resolve_dist_fn(distribution)

    # sample from the given range
    if operation == "add":
        rfi_lim[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device
        )
    elif operation == "abs":
        rfi_lim[env_ids, :] = dist_fn(*distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device)
    elif operation == "scale":
        rfi_lim[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.rfi_lim[env_ids, :] = rfi_lim[env_ids, :]




def resolve_dist_fn(
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    dist_fn = math_utils.sample_uniform

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")

    return dist_fn
