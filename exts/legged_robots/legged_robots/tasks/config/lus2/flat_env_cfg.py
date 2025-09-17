# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import Lus2RoughEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import RigidObject, RigidObjectCfg

@configclass
class Lus2FlatEnvCfg(Lus2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None



class Lus2FlatEnvCfg_PLAY(Lus2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.episode_length_s = 120.0
        self.scene.num_envs = 3
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # increase the light when play
        self.scene.light.spawn = sim_utils.DistantLightCfg(color=(1.65, 1.65, 1.65), intensity=3000.0)

        # feet contact sensor
        self.scene.feet_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*ankle_pitch_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        #filter_prim_paths_expr=["{ENV_REGEX_NS}/World/ground"],
        )

        #setattr(self.scene,"feet_contact_forces",feet_contact_forces)

        # viewer
        self.viewer.origin_type = "envs/env_0/Robot"
        self.viewer.eye = [0.2, 5.4, 0.6]

        # commands
        self.commands.base_velocity.debug_vis=False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None

        # termination
        self.terminations.time_out=None
        self.terminations.base_contact=None
