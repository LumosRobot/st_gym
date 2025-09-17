# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.assets import RigidObject, RigidObjectCfg
import math
from dataclasses import MISSING
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
import legged_robots.tasks.mdp as st_mdp

##
# Pre-defined configs
##
from legged_robots.assets import Lus2_Joint21_CFG, Lus2_Joint27_CFG, Lus2_Joint25_CFG, Lus2_Joint21_CFG_ImplictActuator, Lus2_Joint21_CFG_DelayActuator,Lus2_Joint21_CFG_StDelayActuator

from .amp_mimic_cfg import *

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            texture_scale=(0.25,0.25),
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.4, 0.4, 0.4), intensity=1000.0),
    )


@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.01, 0.1),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=st_mdp.reset_root_state_demo_traj,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=st_mdp.reset_joints_by_demo_traj,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
        },
    )

    # NOTE
    # adding reset event, like EventTerm, to reset ref motion as env reset event happens. note, the reset of ref motion was added in reset_root_state_demo_traj  so far

    """
    Domain randomization
    """
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.75*episode_length_s, episode_length_s),
        params={"velocity_range": {"x": (-0.7, 0.5), "y": (-0.5, 0.5)}},
    )

    #  external disturbance
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.75*episode_length_s, episode_length_s),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-50, 50.0),
            "torque_range": (-30.0, 30.0),
        },
    )

    # body properties
    reset_robot_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link",".*hip.*",".*knee.*",".*shoulder.*"]),
            "mass_distribution_params": (0.95, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_robot_base_com = EventTerm(
        func=st_mdp.randomize_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "distribution_params": ((-0.15, -0.15, -0.15),(0.15, 0.15, 0.15)),
            "operation": "add",
            "distribution": "uniform",
        },
    )


    # joint  and actuators 
    # reset
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
            }
    )

    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 0.15),
            "armature_distribution_params": (0.001, 0.05),
            "lower_limit_distribution_params": (0.0, 0.1),
            "upper_limit_distribution_params": (-0.1, 0.0),
            "operation": "add",
            },
        )

    # actions
    #reset_joint_action_noise_range = EventTerm(
    #    func=st_mdp.randomize_action_noise_range,
    #    mode="reset",
    #    params={
    #        "distribution_params": (0.5, 1.5),
    #        "operation": "scale",
    #        "distribution": "uniform",
    #    },
    #)

    #randomize_rigid_body_collider_offsets = EventTerm(
    #        func = mdp.randomize_rigid_body_collider_offsets,
    #        mode="startup",
    #        params={
    #            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_.*"),
    #            "rest_offset_distribution_params": (0.005, 0.01),
    #            "contact_offset_distribution_params": (0.02,0.05),
    #            "distribution": "uniform",
    #            },
    #        )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    alive_rew = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "alive", "weight": 1.0, "num_steps": 500}
    )

    action_rate_l2 = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_rate_l2", "weight": -1e-1, "num_steps": 500}
    )
    action_smooothness_2 = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_smooothness_2", "weight": -1e-1, "num_steps": 500}
    )
    dof_torques_l2 = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "dof_torques_l2", "weight": -1e-6, "num_steps": 1000}
    )
    dof_acc_l2 = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "dof_acc_l2", "weight": -5e-8, "num_steps": 1000}
    )

    contact_forces = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "contact_forces", "weight": -1.0e-4, "num_steps": 1000}
    )

    #tracking_lin_vel_rew = CurrTerm(
    #    func=mdp.modify_reward_weight, 
    #    params={"term_name": "track_lin_vel_xy_exp", "weight": 5.0, "num_steps": 300}
    #)

    #if upper_joint_name is not None:
    #    tracking_expressive_joint_rew = CurrTerm(
    #    func=mdp.modify_reward_weight, 
    #    params={"term_name": "track_upper_joint_pos_exp", "weight": 150, "num_steps": 500}
    #    )

    #if expressive_link_name is not None:
    #    tracking_expressive_link_rew = CurrTerm(
    #    func=mdp.modify_reward_weight, 
    #    params={"term_name": "track_link_pos_exp", "weight": 40.0, "num_steps": 500}
    #    )

    #command_levels = CurrTerm(
    #    func=st_mdp.command_levels_vel,
    #    params={
    #        "reward_term_name": "track_lin_vel_xy_exp",
    #        "max_curriculum": 1.5,
    #        }
    #)

    # -- internal states
    #update_penalty_curriculum = CurrTerm(
    #    func=st_mdp.update_curriculum,
    #    params={
    #        "penalty_level_down_threshold": 50,
    #        "penalty_level_up_threshold": 900,  # use 110 for hard rough terrain
    #        "penalty_level_degree": 1e-6,
    #        "min_penalty_scale": 0.2,
    #        "max_penalty_scale": 2.0,
    #        "term_names": ["dof_pos_limits", "joint_deviation_hip","joint_deviation_feet","feet_slide","dof_torques_l2","dof_acc_l2","action_rate_l2", "action_smooothness_2", "contact_forces"]
    #    },
    #)
    



@configclass
class Lus2Rewards:
    """Reward terms for the MDP."""
    # Penalize termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-450.0)
    lin_vel_z_l2 = None

    alive = RewTerm(func=mdp.is_alive, weight=5.0)

    # Velocity command tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=track_lin_vel_xy_weight,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=track_ang_vel_z_weight, 
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # Swing feet rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.2,
        },
    )
    # Slide feet penalty 
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    # hip
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_pitch_joint", ".*_hip_roll_joint"])},
    )
    # feet
    joint_deviation_feet = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names= [".*_ankle_.*_joint"])},
    )
    # arms
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*"])},
    )
    # wrist
    if not using_21_joint:
        joint_deviation_wrist = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*_joint"])},
        )
    
    # waist
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.1, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")}
    )
    # Penalize energy cost for locomotion
    energy_cost = RewTerm(
        func=st_mdp.energy_cost,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot",joint_names=["left.*","right.*","torso_joint"])},
        #params={"asset_cfg": SceneEntityCfg("robot",joint_names=["left.*","right.*"])},
    )

    # balance
    #flat_orientation_l2 = RewTerm(
    #    func = mdp.flat_orientation_l2,
    #    weight=-1.0,
    #    params={"asset_cfg": SceneEntityCfg("robot")},
    #)

    feet_parallel_v1 = RewTerm(
        func = st_mdp.feet_parallel_v1,
        weight=feet_parallel_v1_weight,
        params={"asset_cfg": SceneEntityCfg("robot",body_names=[".*_ankle_roll_link"]), "std":0.2},
    )

    # undesired contacts
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*hip_roll_link", ".*hip_yaw_link","torso_link"]), "threshold": 10.0},
    )

    feet_stumble = RewTerm(
        func = st_mdp.feet_stumble,
        weight=feet_stumble_weight,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )


    action_smooothness_2 = RewTerm(
        func=st_mdp.reward_action_smoothness_2,
        weight=-5.0e-2,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-2)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-8)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-4.0e-6)

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, 
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*joint")}
                            )
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*joint"), "soft_ratio": 0.98}
                            )
    dof_torques_limits = RewTerm(func=mdp.applied_torque_limits, weight=-1.0,
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*joint")}
                            )
    # cotact forces
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-5.0e-5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 1200,
        }
    )

    #base_height_l2 = RewTerm(
    #    func= mdp.base_height_l2,
    #    weight=-4.0,
    #    params={"target_height": 1.0},
    #)

    ## Goal commands are only inputs of the policy rather then tracking rewards 
    ## Style goal commands tracking rewards
    if using_amp:
        if style_goal_fields is not None:
            track_style_goal_exp = RewTerm(
                func=st_mdp.track_style_goal_commands_exp,
                weight=track_style_goal_weight,
                params={"command_name": "style_goal_commands", "fields": style_goal_fields, "std": 0.5},
            )

    # Mimic
    # Velocity command tracking rewards
    if using_mimic:
        if upper_joint_name is not None:
            if track_upper_joint_pos_weight !=0:
                track_upper_joint_pos_exp = RewTerm(
                func=st_mdp.track_fields_exp,
                weight=track_upper_joint_pos_weight,
                params={"track_type": "joint_pos",  "field_name": upper_joint_name, "std": 0.4, "tracking_term": "upper_dof_pos"},
                )
            if track_upper_joint_vel_weight !=0:
                track_upper_joint_vel_exp = RewTerm(
                    func=st_mdp.track_fields_exp,
                    weight=track_upper_joint_vel_weight,
                    params={"track_type": "joint_vel",  "field_name": upper_joint_name, "std": 20, "tracking_term": "upper_dof_vel"},
                    )

        if lower_joint_name is not None:
            if track_lower_joint_pos_weight !=0:
                track_lower_joint_pos_exp = RewTerm(
                    func=st_mdp.track_fields_exp,
                    weight=track_lower_joint_pos_weight,
                    params={"track_type": "joint_pos",  "field_name": lower_joint_name, "std": 0.8, "min_std": 0.08, "max_std": 1, "tracking_term": "lower_dof_pos"},
                )
            if track_lower_joint_vel_weight !=0:
                track_lower_joint_vel_exp = RewTerm(
                    func=st_mdp.track_fields_exp,
                    weight=track_lower_joint_vel_weight,
                    params={"track_type": "joint_vel",  "field_name": lower_joint_name, "std": 20, "tracking_term": "lower_dof_vel" },
                    )

        if feet_joint_name is not None:
            if track_feet_joint_pos_weight !=0:
                track_feet_joint_pos_exp = RewTerm(
                    func=st_mdp.track_fields_exp,
                    weight=track_feet_joint_pos_weight,
                    params={"track_type": "joint_pos",  "field_name": feet_joint_name, "std": 0.8, "min_std": 0.1, "max_std": 1.0, "tracking_term": "feet_dof_pos"},
                )
            if track_feet_joint_vel_weight !=0:
                track_feet_joint_vel_exp = RewTerm(
                    func=st_mdp.track_fields_exp,
                    weight=track_feet_joint_vel_weight,
                    params={"track_type": "joint_vel",  "field_name": feet_joint_name, "std": 20, "min_std": 0.1, "max_std":20, "tracking_term": "feet_dof_vel" },
                    )

        if expressive_link_name is not None:
            if track_link_pos_weight !=0:
                track_link_pos_exp = RewTerm( # local frame
                    func=st_mdp.track_fields_exp,
                    weight=track_link_pos_weight,
                    params={"track_type": "link_pos_b",  "field_name": expressive_link_name, "std": 0.015, "tracking_term": "link_pos"},
                )
            if track_link_vel_weight !=0:
                track_link_vel_exp = RewTerm(
                    func=st_mdp.track_fields_exp,
                    weight=track_link_vel_weight,
                    params={"track_type": "link_vel_b",  "field_name": expressive_link_name, "std": 10.0, "tracking_term": "link_vel"},
                )

        # tracking root 
        if track_root_pos_weight !=0:
            track_root_pos_exp = RewTerm(
                func=st_mdp.track_fields_exp,
                weight=track_root_pos_weight,
                params={"track_type": "root_pos_w","std": 0.5, "tracking_term": "root_pos"},
                )
        if track_root_quat_weight !=0:
            track_root_quat_exp = RewTerm(
                func=st_mdp.track_fields_exp,
                weight=track_root_quat_weight,
                params={"track_type": "root_quat_w", "std": 0.5,"tracking_term": "root_quat"},
                )
        if track_root_rotation_weight !=0:
            track_root_rotation_exp = RewTerm(
                 func=st_mdp.track_fields_exp,
                 weight=track_root_rotation_weight,
                 params={"track_type": "root_rotation_w", "std": 0.5,"tracking_term": "root_rotation"},
            )

        if track_root_lin_vel_weight !=0:
            track_root_lin_vel_exp = RewTerm( # world frame
                func=st_mdp.track_fields_exp,
                weight=track_root_lin_vel_weight,
                params={"track_type": "root_lin_vel_w", "std": 8, "tracking_term": "root_lin_vel"},
                )
        if track_root_ang_vel_weight !=0:
            track_root_ang_vel_exp = RewTerm( # world frame
                func=st_mdp.track_fields_exp,
                weight=track_root_ang_vel_weight,
                params={"track_type": "root_ang_vel_w", "std": 8, "tracking_term": "root_ang_vel"},
                )

@configclass
class Lus2ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        #1) proprioception states: observation terms (order preserved)
        base_ang_vel_b = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_actions = ObsTerm(func=mdp.last_action)

        #2) goal states
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        if using_amp:
            if style_goal_fields is not None:
                style_goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "style_goal_commands"})
        if using_mimic:
            if expressive_goal_fields is not None:
                expressive_goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "expressive_goal_commands"})

        #3) for estimate privi info
        if "Encoder" in policy_name:
            p_joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            p_joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            p_last_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            #self.history_length = 5
            #self.flatten_history_dim = False # not flatten


    @configclass
    class CriticPolicyCfg(ObsGroup):
        """Observations for policy group."""

        #1) proprioception states
        # i) root states
        base_pos_w = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_quat_w = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel_b = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel_b = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        # ii) joint and last_action states
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_actions = ObsTerm(func=mdp.last_action)

        # iii) body states 
        if using_mimic:
            if expressive_link_name is not None:
                body_pos_b = ObsTerm(func=st_mdp.body_pos, params={"body_names": expressive_link_name, "root_frame": True}, noise=Unoise(n_min=-0.01, n_max=0.01))
                body_lin_vel_b = ObsTerm(func=st_mdp.body_lin_vel, params={"body_names": expressive_link_name, "root_frame": True}, noise=Unoise(n_min=-0.01, n_max=0.01))

        #2) goal states from the next time frame (future)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        if using_amp:
            if style_goal_fields is not None:
                style_goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "style_goal_commands"})
        if using_mimic:
            if expressive_goal_fields is not None:
                expressive_goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "expressive_goal_commands"})

        #3) privilged obs
        privileged_obs = ObsTerm(func=st_mdp.compute_privileged_observations,
                        params={"asset_cfg": SceneEntityCfg("robot"),
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
                        "terms": ["masses", "contact_forces", "joint_stiffness", "joint_damping", "joint_friction_coeff"], #"joint_armature"],
                        },
                    noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            #self.history_length = 5
            #self.flatten_history_dim = False # not flatten



    @configclass
    class PrivillegedObsCfg(ObsGroup):
        """Observations for policy group."""

        #1) proprioception states
        # i) root states
        privileged_obs = ObsTerm(func=st_mdp.compute_privileged_observations,
                        params={"asset_cfg": SceneEntityCfg("robot"),
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
                        "terms": ["masses", "contact_forces", "joint_stiffness", "joint_damping", "joint_friction_coeff", "joint_armature"],
                        },
                    noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            #self.history_length = 5
            #self.flatten_history_dim = False # not flatten

    """Observation specifications for the MDP."""
    @configclass
    class AmpPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        if "root_pos_x" in style_fields:
            base_pos_x = ObsTerm(func=mdp.base_pos_x, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_pos_y" in style_fields:
            base_pos_y = ObsTerm(func=mdp.base_pos_y, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_pos_z" in style_fields:
            base_pos_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_rot_w" in style_fields:
            base_rot = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_vel_x_b" in style_fields:
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_ang_vel_x_b" in style_fields:
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        if style_joint_name is not None:
            joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot",joint_names=style_joint_name)}, noise=Unoise(n_min=-0.01, n_max=0.01))
            if "left_hip_pitch_joint_dof_vel" in style_fields:
                joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot",joint_names=style_joint_name)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        if style_body_name is not None:
            style_body_pos = ObsTerm(func=st_mdp.body_pos, params={"body_names": style_body_name, "root_frame": True}, noise=Unoise(n_min=-0.01, n_max=0.01))
        #actions = ObsTerm(func=mdp.last_action)
        #height_scan = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True # concate
            self.history_length = amp_obs_frame_num
            self.flatten_history_dim = False # not flatten

    #privileged_obs = PrivillegedObsCfg()

    # observation groups
    if training_teacher: # actor and critic using critic obs which has privi info
        policy: CriticPolicyCfg = CriticPolicyCfg()
    else:
        policy: PolicyCfg = PolicyCfg()
        critic: CriticPolicyCfg = CriticPolicyCfg()


    # amp observation groups
    if "A" in algorithm_name:
        ref_obs: AmpPolicyCfg = AmpPolicyCfg()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        class_type=st_mdp.BaseVelocityCommand,
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1), heading=(-math.pi, math.pi)
        ),
    )

    if style_goal_fields is not None and using_amp:
        style_goal_commands = st_mdp.ExpressiveCommandCfg(
            class_type=st_mdp.StyleCommand,
            asset_name="robot",
            num_commands=len(style_goal_fields),
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )

    if expressive_goal_fields is not None and using_mimic:
        expressive_goal_commands = st_mdp.ExpressiveCommandCfg(
            class_type=st_mdp.ExpressiveCommand,
            asset_name="robot",
            num_commands=len(expressive_goal_fields),
            resampling_time_range=(0.0, 0.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # carefully set this, the illegal_contact indicate the list links contact with anyoyther links
    #bad_contact = DoneTerm(
    #    func=mdp.illegal_contact,
    #    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]), "threshold": 1.0},
    #)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis"), "minimum_height": 0.4}
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis"), "limit_angle": 1.0}
    )
    if using_mimic:
        tracking_lower_dof_error = DoneTerm(
        func=st_mdp.tracking_error_adaptive_termination,
        params={"asset_cfg": SceneEntityCfg("robot"), "error_field": "lower_dof_pos", "min_threshold": 0.3, "max_threshold": 1.5}
    )

        tracking_upper_dof_error = DoneTerm(
        func=st_mdp.tracking_error_adaptive_termination,
        params={"asset_cfg": SceneEntityCfg("robot"), "error_field": "upper_dof_pos", "min_threshold":0.2, "max_threshold":1.5}
    )

    #tracking_root_pos_error = DoneTerm(
    #    func=st_mdp.tracking_error_adaptive_termination,
    #    params={"asset_cfg": SceneEntityCfg("robot"), "error_field": "root_pos", "min_threshold": 0.4, "max_threshold": 2.0,
    #    }
    #)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
    #joint_pos = st_mdp.StJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, position_noise_std=0.01) # 0.01 rad
    #joint_pos = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2)


@configclass
class Lus2RoughEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP settings
    rewards: Lus2Rewards = Lus2Rewards()
    # Basic settings
    observations: Lus2ObservationsCfg = Lus2ObservationsCfg()

    # events
    events: EventCfg = EventCfg()
    # actions
    actions: ActionsCfg = ActionsCfg()
    # commands
    commands: CommandsCfg = CommandsCfg()
    # termination
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # reference motion
    ref_motion: RefMotionCfg = ref_motion_cfg


    def __post_init__(self):
        # general settings
        self.decimation = 10
        self.episode_length_s = episode_length_s
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
            else:
                if self.scene.terrain.terrain_generator is not None:
                    self.scene.terrain.terrain_generator.curriculum = False


        # Scene
        if not using_21_joint:
            self.scene.robot = Lus2_Joint27_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        else:
            #self.scene.robot = Lus2_Joint21_CFG_ImplictActuator.replace(prim_path="{ENV_REGEX_NS}/Robot")
            self.scene.robot = Lus2_Joint21_CFG_StDelayActuator.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.num_envs = num_envs

        #self.scene.feet_contact_forces = ContactSensorCfg(
        #prim_path="{ENV_REGEX_NS}/Robot/.*ankle_link",
        #update_period=0.0,
        #history_length=6,
        #debug_vis=True,
        #filter_prim_paths_expr=["{ENV_REGEX_NS}/World/ground"],
        #)


        self.using_ref_motion_in_actions = False



@configclass
class Lus2RoughEnvCfg_PLAY(Lus2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 120.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # increase the light when play
        self.scene.light.spawn = sim_utils.DistantLightCfg(color=(1.65, 1.65, 1.65), intensity=3000.0)
        # feet contact sensor
        self.scene.feet_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*ankle_roll_link",
        update_period=0.002,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/World/ground"],
        )
        

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.debug_vis=False
        self.commands.expressive_joint_pos.debug_vis=False
        self.commands.expressive_link_pos.debug_vis=False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None

        # termination
        self.terminations.time_out=None
        self.terminations.base_contact=None

        # viewer
        self.viewer.origin_type = "envs/env_0/Robot"
        self.viewer.eye = [0.2, 5.4, 0.6]
        self.viewer.resolution = [1920, 1080]

