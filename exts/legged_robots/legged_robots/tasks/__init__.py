"""Package containing task implementations for various robotic environments."""

import os
import yaml
import pickle
import shutil
from collections import OrderedDict
import toml
import copy

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)


import isaaclab_tasks.utils as isaaclab_utils
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

def parse_env_cfg(task_name: str, args_cli: None) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    
    env_cfg: ManagerBasedRLEnvCfg = isaaclab_utils.parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)

    # override the config by the exiting yaml configuration if sepcified

    ## specify directory for logging experiments
    #log_root_path = os.path.join("logs", "st_rl", agent_cfg.experiment_name)
    #log_root_path = os.path.abspath(log_root_path)
    #print(f"[INFO] Loading experiment from directory: {log_root_path}")
    #resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    #log_dir = os.path.dirname(resume_path)


    if args_cli.cfg_file is not None:   
        if(args_cli.cfg_file=="load_run"): # checking log_file which's value orverwrite cfg.py
            args_cli.cfg_file=os.path.abspath(os.path.join('logs', "st_rl", args_cli.experiment_name, args_cli.load_run, "params"))
        else:
            args_cli.cfg_file=os.path.abspath(os.path.join('logs', "st_rl", args_cli.load_run, "params"))
        print(f"Loading config file from: {args_cli.cfg_file}")
        with open(os.path.join(args_cli.cfg_file, "env.pkl"), "rb") as f:
            obj = pickle.load(f)
            #dict_ = class_to_dict(obj) 
            #update_class_from_dict(env_cfg, dict_, strict= False)
            env_cfg = obj

    # simulation device
    env_cfg.sim.device = args_cli.device
    # number of environments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    return env_cfg


import dataclasses
import os

import dataclasses
import os

def recursive_replace(obj, old="/home/ubuntu", new=None):
    if new is None:
        new = os.environ["HOME"]

    if dataclasses.is_dataclass(obj):
        # loop fields and mutate in place
        for field in dataclasses.fields(obj):
            try:
                value = getattr(obj, field.name)
            except AttributeError:
                continue  # field missing, skip safely
            new_value = recursive_replace(value, old, new)
            setattr(obj, field.name, new_value)
        return obj

    elif isinstance(obj, dict):
        return {k: recursive_replace(v, old, new) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [recursive_replace(v, old, new) for v in obj]

    elif isinstance(obj, str):
        return obj.replace(old, new)

    else:
        return obj




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
                try:
                    setattr(obj, key, val)
                except Exception as e:
                    print(e)
                    import pdb;pdb.set_trace()
        else:
            update_class_from_dict(attr, val)
    return



def is_primitive_type(obj):
    return not hasattr(obj, '__dict__')

def class_to_dict(obj) -> dict:
    if not hasattr(obj,"__dict__") or isinstance(obj, dict):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

