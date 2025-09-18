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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg

import math

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import legged_robots.tasks.locomotion.velocity.mdp as st_mdp

from legged_robots.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from legged_robots.assets import Lus1_CFG, Lus1_Joint25_CFG

from .amp_data_cfg import * 


@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.1, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
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

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # reset
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
            }
    )

    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.05, 0.2),
            "armature_distribution_params": (0.01, 0.1),
            "operation": "abs",
            },

        )


    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            },
        )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    #action_rate = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 500}
    #)
    #joint_vel = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 500}
    #)
    
    tracking_lin_vel_rew = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "track_lin_vel_xy_exp", "weight": 4.0, "num_steps": 300}
    )

    if expressive_joint_name is not None:
        tracking_expressive_rew = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "track_expressive_joint_pos_exp", "weight": 4.0, "num_steps": 500}
        )

    if expressive_link_name is not None:
        tracking_expressive_rew = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "track_expressive_link_pos_weight", "weight": 4.0, "num_steps": 500}
        )

    command_levels = CurrTerm(
        func=st_mdp.command_levels_vel,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "max_curriculum": 1.5,
            }
    )

@configclass
class Lus1Rewards(RewardsCfg):
    """Reward terms for the MDP."""
    # Penalize termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # Velocity command tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # Swing feet rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.2,
        },
    )
    # Slide feet penalty 
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*")}
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    #hip
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    # feet
    joint_deviation_feet = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names= [".*_ankle_.*_joint"])},
    )

    # arms
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])},
    )
    # wrist
    joint_deviation_wrist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*_joint"])},
    )
    
    # waist
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")}
    )
    # Penalize energy cost for locomotion
    #energy_cost = RewTerm(
    #    func=st_mdp.energy_cost,
    #    weight=-2.0e-7,
    #    params={"asset_cfg": SceneEntityCfg("robot",joint_names=["left.*","right.*","torso_joint"])},
    #    #params={"asset_cfg": SceneEntityCfg("robot",joint_names=["left.*","right.*"])},
    #)

    # balance
    flat_orientation_l2 = RewTerm(
        func = mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # undesired contacts
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*hip_pitch_link"), "threshold": 10.0},
    )

    # cotact forces
    #contact_forces = RewTerm(
    #    func=mdp.contact_forces,
    #    weight=-0.0001,
    #    params={
    #        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
    #        "threshold": 1000,
    #    }
    #)

    #base_height_l2 = RewTerm(
    #    func= mdp.base_height_l2,
    #    weight=-4.0,
    #    params={"target_height": 1.0},
    #)

    ## Goal commands are only inputs of the policy rather then tracking rewards 
    ## Style goal commands tracking rewards
    #if style_goal_fields is not None and algorithm_name=="APPO":
    #    track_style_goal_exp = RewTerm(
    #        func=st_mdp.track_style_goal_commands_exp,
    #        weight=4.0,
    #        params={"command_name": "style_goal_commands", "fields": style_goal_fields, "std": 0.5},
    #    )

    # Mimic
    # Velocity command tracking rewards
    if expressive_joint_name is not None:
        track_expressive_joint_pos_exp = RewTerm(
        func=st_mdp.track_expressive_fields_exp,
        weight=track_expressive_joint_pos_weight,
        params={"expressive_type": "joint_pos",  "field_name": expressive_joint_name, "std": 1.2},
        )

    if expressive_link_name is not None:
        track_expressive_link_pos_exp = RewTerm(
        func=st_mdp.track_expressive_fields_exp,
        weight=track_expressive_link_pos_weight,
        params={"expressive_type": "link_pos",  "field_name": expressive_link_name, "std": 1.1},
        )



@configclass
class Lus1ObservationsCfg(ObservationsCfg):

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        if style_goal_fields is not None and algorithm_name=="APPO":
            style_goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "style_goal_commands"})
        if expressive_goal_fields is not None:
            expressive_goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "expressive_goal_commands"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            #self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()


    @configclass
    class CriticPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        if style_goal_fields is not None and algorithm_name=="APPO":
            style_goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "style_goal_commands"})
        if expressive_goal_fields is not None:
            expressive_goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "expressive_goal_commands"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            #self.history_length = 5

    # observation groups
    critic: CriticPolicyCfg = CriticPolicyCfg()


    """Observation specifications for the MDP."""
    @configclass
    class AmpPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        #base_pos = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_pos_x" in style_fields:
            base_pos_x = ObsTerm(func=mdp.base_pos_x, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_pos_y" in style_fields:
            base_pos_y = ObsTerm(func=mdp.base_pos_y, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_pos_z" in style_fields:
            base_pos_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_rot_w" in style_fields:
            base_rot = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_vel_x" in style_fields:
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        if "root_ang_vel_x" in style_fields:
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        #velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        if style_joint_name is not None and algorithm_name=="APPO":
            joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot",joint_names=style_joint_name)}, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot",joint_names=style_joint_name)}, noise=Unoise(n_min=-0.01, n_max=0.01))
        #joint_default_pos = ObsTerm(func=st_mdp.joint_default_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        if style_body_name is not None and algorithm_name=="APPO":
            style_body_pos = ObsTerm(func=st_mdp.style_body_pos, params={"body_names":style_body_name}, noise=Unoise(n_min=-0.01, n_max=0.01))
        #actions = ObsTerm(func=mdp.last_action)
        #height_scan = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True # concate
            self.history_length = amp_obs_frame_num
            self.flatten_history_dim = False # not flatten

    # amp observation groups
    if algorithm_name=="APPO":
        amp_policy: AmpPolicyCfg = AmpPolicyCfg()



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
            lin_vel_x=(0, 0.2), lin_vel_y=(0.0, 0.1), ang_vel_z=(-0.1, 0.1), heading=(-math.pi, math.pi)
        ),
    )

    if style_goal_fields is not None and algorithm_name=="APPO":
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

    if expressive_goal_fields is not None:
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
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link",".*hip_pitch_link"]), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "limit_angle": 1.1}
    )
    
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class Lus1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Lus1Rewards = Lus1Rewards()
    # Basic settings
    observations: Lus1ObservationsCfg = Lus1ObservationsCfg()

    # events
    events: EventCfg = EventCfg()
    # actions
    actions: ActionsCfg = ActionsCfg()
    # commands
    commands: CommandsCfg = CommandsCfg()
    # termination
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        #self.scene.robot = Lus1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = Lus1_Joint25_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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


        # additional objects for visualization



        # Randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        #self.events.base_external_force_torque = None
        #self.events.push_robot = None
        #self.events.add_base_mass = None

        # Terminations


        # Rewards
        self.rewards.dof_torques_l2.weight = -1.0e-8
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7


@configclass
class Lus1RoughEnvCfg_PLAY(Lus1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
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

        # viewer
        self.viewer.origin_type = "envs/env_0/Robot"
        self.viewer.eye = [0.2, 5.4, 0.6]
        self.viewer.resolution = [1920, 1080]

