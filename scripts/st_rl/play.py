"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import numpy
import glob

# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger("play")
from isaaclab.utils.math import quat_rotate_inverse



from isaaclab.app import AppLauncher
from collections import OrderedDict

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
#parser.add_argument("--log", action="store_true", default=False, help="Record log during training.")
parser.add_argument("--log_length", type=int, default=200, help="Length of the recorded log (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--cfg_file", type=str, default=None, help="a config file used to override the cfg class")
parser.add_argument("--play_demo_traj", action="store_true", default=False, help="play a demo trajectory")

# append RSL-RL cli arguments
cli_args.add_st_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
logger.info(f"Hydr args: {hydra_args}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

import omni.log
from isaaclab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse

from st_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
#from isaacsim.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from legged_robots.tasks.utils.wrappers.st_rl import (
    export_policy_as_jit,
    export_policy_as_onnx,
)
#from st_rl.utils.metrics import StRlMetricEnvWrapper
from legged_robots.tasks.utils.wrappers.st_rl import StRlVecEnvWrapper, StRlMetricEnvWrapper

# Import extensions to set up environment tasks
import legged_robots.tasks  # noqa: F401
from legged_robots.tasks  import parse_env_cfg, recursive_replace# noqa: F401
from legged_robots.tasks.config.lus2.amp_mimic_cfg import all_joint_names
from legged_robots.assets import Lus2_Joint21_CFG, Lus2_Joint27_CFG, Lus2_Joint25_CFG


#@hydra_task_config(args_cli.task, "st_rl_cfg_entry_point")
def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, args_cli)
    # 替换整个 env_cfg
    env_cfg = recursive_replace(env_cfg, old="/home/ubuntu", new=os.environ["HOME"])

    env_cfg.scene.num_envs=int(args_cli.num_envs)
    # set this robot accoring to to situation
    env_cfg.scene.robot = Lus2_Joint21_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # remove random pushing
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.update_curriculum = None
    env_cfg.curriculum = None

    # termination
    env_cfg.terminations.time_out=None
    env_cfg.terminations.base_contact=None

    # ref motion
    if getattr(env_cfg, "ref_motion", None) is not None:
        env_cfg.ref_motion.trajectory_num=int(args_cli.num_envs)
        env_cfg.ref_motion.ref_length_s=None
        env_cfg.ref_motion.device = args_cli.device
        specify_init_values = {key+"_dof_pos": 0.0 for key in all_joint_names}
        specify_init_values["root_rot_x"] = 0
        specify_init_values["root_rot_y"] = 0
        specify_init_values["root_rot_z"] = 0
        specify_init_values["root_rot_w"] = 1
        specify_init_values["root_pos_z"] = 0.86
        specify_init_values["left_hip_pitch_joint_dof_pos"] = -0.37
        specify_init_values["right_hip_pitch_joint_dof_pos"] = -0.37
        specify_init_values["left_knee_joint_dof_pos"] = 0.74
        specify_init_values["right_knee_joint_dof_pos"] = 0.74
        specify_init_values["left_ankle_pitch_joint_dof_pos"] = -0.37
        specify_init_values["right_ankle_pitch_joint_dof_pos"] = -0.37
        specify_init_values["left_shoulder_roll_joint_dof_pos"] = 0.25
        specify_init_values["right_shoulder_roll_joint_dof_pos"] = -0.25
        specify_init_values["left_elbow_joint_dof_pos"] = 1.2
        specify_init_values["right_elbow_joint_dof_pos"] = 1.2

        env_cfg.ref_motion.specify_init_values = specify_init_values
        env_cfg.ref_motion.motion_files = glob.glob(f"{os.getenv('HOME')}/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/dance*fps*")
        if len(env_cfg.ref_motion.motion_files) < 1:
            env_cfg.ref_motion.motion_files = glob.glob(f"{os.getenv('HOME')}/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/dance*fps*")
            print(f"The ref motion for training do not exist, change to use {env_cfg.ref_motion.motion_files}")
        if not os.path.isfile(env_cfg.ref_motion.motion_files[0]):
            env_cfg.ref_motion.motion_files = glob.glob(os.path.join(os.getenv("HOME"),env_cfg.ref_motion.motion_files[0][env_cfg.ref_motion.motion_files[0].find("workspace"):]))
        print(f"Ref motion files: {env_cfg.ref_motion.motion_files}")
            
        env_cfg.ref_length_s = None # s

        if args_cli.play_demo_traj:
            env_cfg.ref_motion.style_fields = env_cfg.ref_motion.init_state_fields

    #args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_st_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "st_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    logger.info(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 100,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        logger.info("[INFO] Recording videos during playing.")
        logger.info_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    env = StRlVecEnvWrapper(env)
    # wrap around environment for metrics
    strl_kwargs = {
        "log_dir": os.path.join(log_dir,"metrics"),
        "enable_logger":True,
    }
    env = StRlMetricEnvWrapper(env,**strl_kwargs)
    # robot joint names
    if args_cli.video:
        joint_names = env.unwrapped.scene['robot'].joint_names
    else:
        joint_names = env.unwrapped.scene['robot'].joint_names
    logger.info(f"[Info] Robot joint names: {joint_names}")

    # load the trained model
    logger.info(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, map_location={"cuda:1":agent_cfg.device})

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ROS interface
    #import threading
    #import rclpy
    #from sensor_msgs.msg import JointState
    #from
    #rclpy.init()
    #node = rclpy.create_node('play_subscriber')
    #pub = node.create_publisher(JointState, 'joint_command', 10)

    ## Spin in a separate thread
    #thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    #thread.start()

    #joint_state_position = JointState()
    #joint_state_velocity = JointState()

    #joint_state_position.name = ["joint1", "joint2","joint3"]
    #joint_state_velocity.name = ["wheel_left_joint", "wheel_right_joint"]
    #joint_state_position.position = [0.2,0.2,0.2]
    #joint_state_velocity.velocity = [20.0, -20.0]

    #rate = node.create_rate(10)
    #try:
    #    while rclpy.ok():
    #        pub.publish(joint_state_position)
    #        pub.publish(joint_state_velocity)
    #        rate.sleep()
    #except KeyboardInterrupt:
    #    pass
    #rclpy.shutdown()
    #thread.join()

    if args_cli.play_demo_traj:
        # Assume: num_bodies = 12 (example), modify as needed
        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        # Create the markers configuration
        # This creates two marker prototypes, "marker1" and "marker2" which are spheres with a radius of 1.0.
        # The color of "marker1" is red and the color of "marker2" is green.
        cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/testMarkers",
                markers={
                    #"cone": sim_utils.ConeCfg(
                    #    radius=0.2,
                    #    height=0.5,
                    #    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                    #    ),
                    "marker1": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "marker2": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "arrow_x": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(1.0, 0.5, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                    ),
                    }
                )
        # Create the markers instance
        # This will create a UsdGeom.PointInstancer prim at the given path along with the marker prototypes.
        marker = VisualizationMarkers(cfg)
        

    # export policy to onnx/jit
    EXPORT_MODEL = True
    if EXPORT_MODEL:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        #export_policy_as_jit(
        #    ppo_runner.alg.actor_critic, ppo_runner.alg.obs_normalizer, path=export_model_dir, filename="policy.pt"
        #)
        if hasattr(ppo_runner.alg.actor_critic,"encoders"):
            from st_rl.utils.utils import get_obs_slice
            from legged_robots.tasks.config.lus2.amp_mimic_cfg import obs_segments, encoder_component_names
            obs_slices = [get_obs_slice(obs_segments, name) for name in encoder_component_names]
        else:
            obs_slices = None
        export_policy_as_onnx(
            ppo_runner.alg.actor_critic, normalizer=ppo_runner.alg.obs_normalizer, path=export_model_dir, filename="policy.onnx",
            obs_slices=obs_slices,
        )

    # create tele-operation
    if args_cli.teleop_device.lower() == "keyboard":
        sensitivity = 0.5
        teleop_interface = Se3Keyboard(
            pos_sensitivity= sensitivity, rot_sensitivity= sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")

    # add teleoperation key for env reset
    teleop_interface.add_callback("R", env.reset)
    teleop_interface.add_callback("L", env.start_logging)
    teleop_interface.add_callback("K", env.finish_logging)
    # logger.info helper for keyboard
    logger.info(teleop_interface)

    # reset environment
    obs, extras = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        timestep += 1
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            if not args_cli.play_demo_traj:
                obs, _, _, extras = env.step(actions)
            # get commands from teleop
            if not hasattr(env.cfg, "ref_motion"):
                delta_pose, gripper_command = teleop_interface.advance()
                delta_pose = torch.tensor(delta_pose.astype("float32"))
                env.commands[:,:3] = delta_pose[:3]

            if args_cli.play_demo_traj:
                traj_id = (timestep % env.unwrapped.ref_motion.root_pos_w.shape[0])
                dof_num=len(joint_names)
                root_states = extras["ref_motion"][traj_id, :13].unsqueeze(0)
                #i) using ref motion from extras
                #dof_pos = extras["ref_motion"][traj_id, 13:13+dof_num].unsqueeze(0)
                #dof_vel = extras["ref_motion"][traj_id, 13+dof_num:13+2*dof_num].unsqueeze(0)

                #ii) using ref motion directly from ref_motion
                dof_pos_index = [env.unwrapped.ref_motion.trajectory_fields.index(key1+"_dof_pos") for key1 in joint_names]
                dof_pos = env.unwrapped.ref_motion.data[:,dof_pos_index]
                dof_vel_index = [env.unwrapped.ref_motion.trajectory_fields.index(key1+"_dof_vel") for key1 in joint_names]
                dof_vel = env.unwrapped.ref_motion.data[:,dof_vel_index]

                env.unwrapped.scene._articulations['robot'].write_root_state_to_sim(root_states)
                env.unwrapped.scene._articulations['robot'].write_joint_state_to_sim(dof_pos, dof_vel)

                # get robot body states
                vis_link_names = ["left_hip_pitch_link","right_elbow_link"]
                body_ids = [env.unwrapped.scene["robot"].body_names.index(name) for name in vis_link_names]
                body_pos_w = env.unwrapped.scene["robot"].data.body_pos_w[:, body_ids, :] - env.unwrapped.scene.env_origins.unsqueeze(1)

                s_root_pos_w = env.unwrapped.scene["robot"].data.root_pos_w
                s_root_quat_w = env.unwrapped.scene["robot"].data.root_quat_w # (envs, 4)

                body_pos_b = body_pos_w - s_root_pos_w.unsqueeze(1)         # (envs, bodies, 3)
                root_quat_w_expanded = s_root_quat_w.unsqueeze(1).expand(-1, body_pos_w.shape[1], -1).reshape(-1, 4) # (envs_bodies,4)
                body_pos_b = quat_rotate_inverse(root_quat_w_expanded, body_pos_b.reshape(-1,3))  # (envs*bodies, 3)
                body_pos_w = body_pos_w.squeeze(0)

                # getting ref motion data
                root_pos_w = env.unwrapped.ref_motion.root_pos_w
                root_quat_w = env.unwrapped.ref_motion.root_quat_w
                root_lin_vel_w = env.unwrapped.ref_motion.root_lin_vel_w
                if (hasattr(env.unwrapped.ref_motion, "data")):
                    expre_field_index = [env.unwrapped.ref_motion.trajectory_fields.index(key1+key2) for key1 in vis_link_names for key2 in ["_pos_x_w", "_pos_y_w", "_pos_z_w"]]
                    link_pos_w = env.unwrapped.ref_motion.data[:,expre_field_index].reshape(-1,3)
                if (hasattr(env.unwrapped.ref_motion, "data")):
                    expre_field_index = [env.unwrapped.ref_motion.trajectory_fields.index(key1+key2) for key1 in vis_link_names for key2 in ["_pos_x_b", "_pos_y_b", "_pos_z_b"]]
                    link_pos_b = env.unwrapped.ref_motion.data[:,expre_field_index].reshape(-1,3)

                root_lin_vel_w_norm = root_lin_vel_w/torch.norm(root_lin_vel_w, p=2, dim=1, keepdim=True)
                marker_translations = torch.cat([body_pos_w[0,:].unsqueeze(0), link_pos_w[0,:].unsqueeze(0)])
                marker_orientations = root_quat_w
                marker_indices = [0, 1] #list(range(len(vis_link_names)))
                marker.visualize(marker_indices=marker_indices, translations=marker_translations, orientations=marker_orientations)

                obs, _, _, extras = env.step(dof_pos)

            if env.start_log:
                env.log_update()
                if env.finish_log:
                    env.log_save()
                    logger.info("[INFO] Log saved.")
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
