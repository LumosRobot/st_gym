import torch

from legged_robots.tasks.locomotion.velocity.mdp.observations import *
from isaaclab.utils.math import quat_rotate


def test_body_transform_consistency(env, body_names):
    pos_root = body_pos_w(env, body_names=body_names, root_frame=True).reshape(-1, len(body_names), 3)
    vel_root = body_lin_vel_w(env, body_names=body_names, root_frame=True).reshape(-1, len(body_names), 3)

    asset = env.scene["robot"]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w

    # Reconstruct world
    pos_world_recon = quat_rotate(root_quat, pos_root) + root_pos.unsqueeze(1)
    vel_world_recon = quat_rotate(root_quat, vel_root)

    # True world-frame
    pos_world_true = body_pos(env, body_names=body_names, root_frame=False).reshape(-1, len(body_names), 3)
    vel_world_true = body_lin_vel(env, body_names=body_names, root_frame=False).reshape(-1, len(body_names), 3)

    pos_match = torch.allclose(pos_world_true, pos_world_recon, atol=1e-5)
    vel_match = torch.allclose(vel_world_true, vel_world_recon, atol=1e-5)

    print("Position transform correct:", pos_match)
    print("Velocity transform correct:", vel_match)
    if not pos_match:
        print("Position diff:", (pos_world_true - pos_world_recon).abs().max())
    if not vel_match:
        print("Velocity diff:", (vel_world_true - vel_world_recon).abs().max())

