
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def body_pos(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    body_names: list = ["left_elbow_link", "right_elbow_link"],
    root_frame: bool = False,
) -> torch.Tensor:
    """Returns the world-frame positions of specified bodies."""

    asset: RigidObject = env.scene[asset_cfg.name]
    try:
        body_ids = [asset.body_names.index(name) for name in body_names]
    except ValueError as e:
        raise ValueError(f"One or more body names not found in asset.body_names: {body_names}") from e

    body_pos_w = asset.data.body_pos_w[:, body_ids, :] - env.scene.env_origins.unsqueeze(1)

    # Get root pose and orientation
    if root_frame:
        root_pos_world = asset.data.root_pos_w  # (envs, 3)
        root_quat_world = asset.data.root_quat_w # (envs, 4)
        body_pos_w = body_pos_w - root_pos_world.unsqueeze(1)         # (envs, bodies, 3)
        root_quat_world_expanded = root_quat_world.unsqueeze(1).expand(-1, len(body_ids), -1).reshape(-1, 4) # (envs_bodies,4)
        body_pos_b = quat_rotate_inverse(root_quat_world_expanded, body_pos_w.reshape(-1,3))  # (envs*bodies, 3)
        return body_pos_b.reshape(-1, len(body_ids)*3)
    else:
        return body_pos_w.reshape(-1, len(body_ids)*3)



def body_lin_vel(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    body_names: list = ["left_elbow_link", "right_elbow_link"],
    root_frame: bool = False,
) -> torch.Tensor:
    """Returns the world-frame linear velocities of specified bodies."""
    asset: RigidObject = env.scene[asset_cfg.name]
    try:
        body_ids = [asset.body_names.index(name) for name in body_names]
    except ValueError as e:
        raise ValueError(f"One or more body names not found in asset.body_names: {body_names}") from e

    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_ids, :]

    if root_frame:
        root_quat_world = asset.data.root_quat_w # (envs, 4)
        root_quat_world_expanded = root_quat_world.unsqueeze(1).expand(-1, len(body_ids), -1).reshape(-1, 4) # (envs_bodies,4)
        body_lin_vel_b = quat_rotate_inverse(root_quat_world_expanded, body_lin_vel_w.reshape(-1,3)) # (env*bodies,3)
        return body_lin_vel_b.reshape(-1, len(body_ids)*3)
    else:
        return body_lin_vel_w.reshape(-1,len(body_ids)*3)



###asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
#    """The generated command from command term in the command manager with the given name."""
#    return env.command_manager.get_command(command_name)


#def privileged_ref_motion(
#    env: ManagerBasedEnv, 
#    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
#    field_num: int,
#) -> torch.Tensor:
#    """Root linear velocity in the asset's root frame."""
#    asset: RigidObject = env.scene[asset_cfg.name]
#    if hasattr(env,"ref_motion"):
#        return env.ref_motion.privileged_ref_motion
#    else:
#        return torch.zeros(field_num)


def compute_privileged_observations(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
        terms = ["masses", "contact_forces", "joint_stiffness", "joint_damping", "joint_friction_coeff"], #"joint_armature"],
        ):

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:,sensor_cfg.body_ids].reshape(-1,len(sensor_cfg.body_ids)*3)

    asset: Articulation = env.scene[asset_cfg.name]

    masses = asset.root_physx_view.get_masses()

    joint_stiffness = asset.data.joint_stiffness
    joint_damping = asset.data.joint_damping
    joint_friction_coeff = asset.data.joint_friction_coeff
    joint_armature = asset.data.joint_armature
    privileged_obs_dict ={}

    for key in terms:
        if key in locals():
            privileged_obs_dict[key] = locals()[key].to(env.device)

    #privileged_obs_dict = {
    #    "masses": masses.to(env.device),
    #     "contact_forces": contact_forces,
    #     "joint_stiffness": joint_stiffness,
    #     "joint_damping": joint_damping,
    #     "joint_friction_coeff": joint_friction_coeff,
    #     "joint_armature": joint_armature
    #}
    privileged_obs = torch.cat(
        [tensor for tensor in privileged_obs_dict.values()],
        dim=-1,
    )
    return privileged_obs
