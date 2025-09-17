from __future__ import annotations

import argparse
from typing import TYPE_CHECKING
import os
import pickle
import copy


if TYPE_CHECKING:
    from isaaclab_rl.st_rl import RslRlOnPolicyRunnerCfg


def add_st_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("st_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_st_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """

    # load the default configuration
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    strl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "st_rl_cfg_entry_point")


    # override the config by the exiting yaml configuration if sepcified
    if args_cli.cfg_file is not None:   
        args_cli.cfg_file=os.path.abspath(os.path.join('logs', "st_rl", args_cli.experiment_name, args_cli.load_run, "params"))
        print(f"Loading agent config file from: {args_cli.cfg_file}")
        with open(os.path.join(args_cli.cfg_file, "agent.pkl"), "rb") as f:
            obj = pickle.load(f)
            #dict_ = class_to_dict(obj) 
            #update_class_from_dict(env_cfg, dict_, strict= False)
            #import pdb;pdb.set_trace()
            strl_cfg = obj
    # override cfg from args (if specified)
    strl_cfg = update_st_rl_cfg(strl_cfg, args_cli)

    return strl_cfg


def update_st_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg



def update_class_from_dict(obj, dict_, strict= False):
    """ If strict, attributes that are not in dict_ will be removed from obj """
    if strict:
        attr_names = [n for n in obj.__dict__.keys() if not (n.startswith("__") and n.endswith("__"))]
        for attr_name in attr_names:
            if not attr_name in dict_:
                delattr(obj, attr_name)
    for key, val in dict_.items():
        attr = getattr(obj, key, None)
        if attr is None or is_primitive_type(attr):
            if isinstance(val, dict):
                setattr(obj, key, copy.deepcopy(val))
                update_class_from_dict(getattr(obj, key), val)
            else:
                setattr(obj, key, val)
        else:
            update_class_from_dict(attr, val)
    return


