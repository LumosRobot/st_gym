if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

def test_track_expressive_fields_exp(env: ManagerBasedRLEnv):
    """Unit test for track_expressive_fields_exp function."""
    asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]

    # Assume ref_motion already exists in env
    assert hasattr(env, "ref_motion"), "env must have ref_motion for this test."

    std = 0.5
    tol = 1e-4  # tolerance for numerical equality
    passed = True

    # 1. Joint Position (when goal == current)
    joint_names = asset.data.joint_names[:2]
    idx = [asset.data.joint_names.index(name) for name in joint_names]
    env.ref_motion.expressive_joint_pos = asset.data.joint_pos[:, idx].clone()
    reward = track_expressive_fields_exp(env, std, "joint_pos", joint_names, asset_cfg)
    if not torch.allclose(reward, torch.ones_like(reward), atol=tol):
        print("❌ joint_pos test failed: expected reward ~1.0")
        passed = False

    # 2. Joint Velocity (with noise)
    env.ref_motion.expressive_joint_vel = asset.data.joint_vel[:, idx].clone() + 0.1
    reward = track_expressive_fields_exp(env, std, "joint_vel", joint_names, asset_cfg)
    if not torch.all(reward < 1.0):
        print("❌ joint_vel test failed: expected reward < 1.0 for perturbed goal")
        passed = False

    # 3. Link Position in Root Frame
    body_names = asset.body_names[:2]
    body_ids = [asset.body_names.index(name) for name in body_names]
    body_pos_w = asset.data.body_pos_w[:, body_ids, :] - env.scene.env_origins.unsqueeze(1)
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    pos_b = quat_rotate_inverse(root_quat, body_pos_w - root_pos.unsqueeze(1))
    env.ref_motion.expressive_link_pos_b = pos_b.reshape(env.num_envs, -1)
    reward = track_expressive_fields_exp(env, std, "link_pos_b", body_names, asset_cfg)
    if not torch.allclose(reward, torch.ones_like(reward), atol=tol):
        print("❌ link_pos_b test failed: expected reward ~1.0")
        passed = False

    # 4. Root Angular Velocity (zero test)
    env.ref_motion.expressive_root_ang_vel_b = asset.data.root_ang_vel_b.clone()
    reward = track_expressive_fields_exp(env, std, "root_ang_vel_b", field_name=[], asset_cfg=asset_cfg)
    if not torch.allclose(reward, torch.ones_like(reward), atol=tol):
        print("❌ root_ang_vel_b test failed")
        passed = False

    if passed:
        print("✅ All tests passed for track_expressive_fields_exp")


def test_feet_parallel_l1_rewards(env: ManagerBasedRLEnv):

    asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]

    rewards = feet_parallel_v1(env, asset_cfg)
    print(f" feet parallel l1 rewards: {rewards})
