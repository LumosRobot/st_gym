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
def get_env(env_cfg: DictConfig, agent_cfg:DictConfig):
   env = MujocoSimEnv(env_cfg)
   return env

ENV = get_env()

class TestSim2Mujoco(unittest.TestCase):
    """Tests for WBCMujoco wrapper"""

    def setUp(self):
        # Create a WBCMujoco object for testing
        # NOTE: We load the scene.xml to include the floor used for contact forces testing

        self.env=ENV

        self.model_path = get_data_path("/home/thomas/workspace/lumos_ws/lumos_rl_gym/resources/robots/lus2/mjcf/lus2_joint25.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        logger.info(f"xml model path: {self.model_path}")
        self.sim_dt = 0.005
        
        # mj model
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # mj_data
        self.enable_viewer = False
        self.device = torch.device("cpu")

    def test_mojoco(self):
        """
        Run the Mujoco simulation using the provided policy and configuration.
    
        Args:
            policy: The policy used for controlling the simulation.
            cfg: The configuration object containing simulation settings.
    
        Returns:
            None
        """
    
        """Play with RSL-RL agent."""
        # loading mujoco model and data
        mujoco.mj_step(self.mj_model, self.mj_data)
        viewer = mujoco_viewer.MujocoViewer(self.mj_model, self.mj_data)
    
        sim_duration=10
        
        for _ in tqdm(range(int(sim_duration /dt)), desc="Simulating..."):
            mujoco_joint_names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.mj_model.njnt)][1:]
            joint_num = self.mj_model.njnt-1
    
            target_q = np.zeros((joint_num),dtype=np.double)
            target_dq = np.zeros((joint_num),dtype=np.double)
            kps = 100*np.ones((joint_num),dtype=np.double)
            
            kds = 2*np.ones((joint_num),dtype=np.double)
    
            #tau = pd_control(target_q, self.mj_data.qpos[7:], kps, target_dq, mj_data.qvel[6:], kds)  # Calc torques
            #mj_data.ctrl = tau
            
            mujoco.mj_step(self.mj_model, self.mj_data)
            viewer.render()
            
        viewer.close()

    def test_gvec(self):
        quat = np.array([0.0, 0.707, 0.0, 0.707]).astype(np.double)
        r = R.from_quat(quat)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        logger.info(f"gvec: {gvec}")




if __name__ == "__main__":
    unittest.main()
