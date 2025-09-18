import numpy as np
import torch
import unittest
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

import time
import numpy as np
import mujoco #, mujoco_viewer
import mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

import glob
import re
import torch
import onnx
import onnxruntime as ort


# local imports
import argparse
import sys
import os
import numpy
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'st_rl')))
from sim2mujoco import MujocoSimEnv, InferenceRunner, args_cli, hydra_mj_config


# Configure the logging system
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define the log message format
)
# Create a logger object
logger = logging.getLogger(__file__)


args_cli.experiment_name="lus2_flat"
args_cli.load_run="tag_il_gesture_joint27_2"
args_cli.task="Isaac-Velocity-Flat-Lus2-vst"

@hydra_mj_config(args_cli)
def get_runnr(env_cfg: DictConfig, agent_cfg:DictConfig):
   env = MujocoSimEnv(env_cfg)
   runner=InferenceRunner(env, agent_cfg, args_cli)
   return runner


class TestSimInference(unittest.TestCase):
    """Tests for WBCMujoco wrapper"""

    def setUp(self):
        # Create a WBCMujoco object for testing
        # NOTE: We load the scene.xml to include the floor used for contact forces testing


        args_cli.experiment_name="lus2_flat"
        args_cli.load_run="tag_il_gesture_joint27_2"
        args_cli.task="Isaac-Velocity-Flat-Lus2-vst"

        self.runner = RUNNER

        obs_data_path = "/home/thomas/workspace/lumos_ws/lumos_controller/obs.csv" # Adjust column range if needed
        action_data_path='/home/thomas/workspace/lumos_ws/lumos_controller/actions.csv' # Adjust column range if needed
        import pandas as pd
        action_data = pd.read_csv(action_data_path,delimiter='\t').values
        obs_data = pd.read_csv(obs_data_path,delimiter='\t').values

        logger.info(f"action data shape: {action_data.shape}")
        logger.info(f"obs data shape: {obs_data.shape}")


    def test_policy(self):

        policy = self.runner.get_inference_policy()

        obs = self.obs_data
        actions = self.action_data

        eval_actions =[]
        for idx in range(obs.shape[0]):
            eval_actions.append(policy(obs[idx,:].reshape(1,-1)))

        eval_actions = np.array(eval_actions)

        rmse = np.sqrt(np.mean((eval_actions - actions) ** 2))

        logger.info("eval action error: {rmse}")
        

if __name__ == "__main__":
    st_run = get_runnr()
    import pdb;pdb.set_trace()
    unittest.main()
