# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-Lus1-vst",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Lus1RoughEnvCfg",
        "st_rl_cfg_entry_point": f"{agents.__name__}.st_rl_ppo_cfg:Lus1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-Lus1-Play-vst",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Lus1RoughEnvCfg_PLAY",
        "st_rl_cfg_entry_point": f"{agents.__name__}.st_rl_ppo_cfg:Lus1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Lus1-vst",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Lus1FlatEnvCfg",
        "st_rl_cfg_entry_point": f"{agents.__name__}.st_rl_ppo_cfg:Lus1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)



gym.register(
    id="Isaac-Velocity-Flat-Lus1-Play-vst",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Lus1FlatEnvCfg_PLAY",
        "st_rl_cfg_entry_point": f"{agents.__name__}.st_rl_ppo_cfg:Lus1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
