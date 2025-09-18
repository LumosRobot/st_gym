# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`UNITREE_GO1_CFG`: Unitree Go1 robot with actuator net model for the legs
* :obj:`UNITREE_GO2_CFG`: Unitree Go2 robot with DC motor model for the legs
* :obj:`Lus1_CFG`: H1 humanoid robot
* :obj:`G1_CFG`: G1 humanoid robot
* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg,DelayedPDActuatorCfg, RemotizedPDActuatorCfg, DelayedPDActuator
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
import torch
from collections.abc import Sequence
from isaaclab.utils.types import ArticulationActions



class ActionScaleCalculator:
    """计算关节动作缩放比例的类"""
    
    def __init__(self, scale_factor=0.25):
        self.scale_factor = scale_factor
    
    def calculate(self, cfg_obj):
        """
        计算指定配置的动作缩放比例
        
        参数:
            cfg_obj: 配置对象，应包含actuators字典属性
            
        返回:
            包含关节和对应缩放比例的字典
        """
        result = {}
        for a in cfg_obj.actuators.values():
            e = a.effort_limit_sim  # 力矩限制
            s = a.stiffness         # 刚度
            names = a.joint_names_expr  # 关节名称
            
            # 处理标量值转换为字典形式
            if not isinstance(e, dict):
                e = {n: e for n in names}
            if not isinstance(s, dict):
                s = {n: s for n in names}
            
            # 计算每个关节的ACTION_SCALE
            for n in names:
                if n in e and n in s and s[n]:  # 确保刚度不为零
                    result[n] = self.scale_factor * e[n] / s[n]
        
        return result

class StDelayedPDActuator(DelayedPDActuator):
    """Ideal PD actuator with angle-dependent torque limits.

    This class extends the :class:`DelayedPDActuator` class by adding angle-dependent torque limits to the actuator.
    The torque limits are applied by querying a lookup table describing the relationship between the joint angle
    and the maximum output torque. The lookup table is provided in the configuration instance passed to the class.

    The torque limits are interpolated based on the current joint positions and applied to the actuator commands.
    """

    def __init__(
        self,
        cfg: DelayedPDActuatorCfg,
        joint_names: list[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,

    ):
        # call the base method and set default effort_limit and velocity_limit to inf
        super().__init__(
            cfg, joint_names, joint_ids, num_envs, device, stiffness, damping, armature, friction, torch.inf, torch.inf
        )
        # define remotized joint torque limit
        print(f"StDelayedPDActuator effort strength: {cfg.effort_strength}")
        self._effort_strength = math_utils.sample_uniform(*cfg.effort_strength, (num_envs, self.num_joints), device=device)


    """
    Operations.
    """

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)
        computed_effort = control_action.joint_efforts*self._effort_strength
        applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = applied_effort

        return control_action



from isaaclab.utils import configclass

@configclass
class StDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = StDelayedPDActuator

    effort_strength: tuple[float, float] = (1.0, 1.0)


def get_kpkd(motor_type):
    damp_ratio = 1.2
    freq = 10  # rad/s
    if motor_type == "6030":
        motor_ratio = 30
        I_motor = 14e-6
    elif motor_type == "6005":   # "else if" → "elif"
        motor_ratio = 5           # fixed typo "motor_ratiio"
        I_motor = 14e-6
        freq = 15  # rad/s
        damp_ratio = 1.1
    elif motor_type == "10424":  # missing colon
        motor_ratio = 24
        I_motor = 132e-6
    else:
        raise ValueError(f"Unknown motor_type: {motor_type}")

    w_n = 2*3.1425* freq
    I_j = motor_ratio**2 * I_motor
    #print(f"I j is {I_j}, I_motor is {I_motor}")

    kp = I_j * w_n**2
    kd = 2 * damp_ratio * I_j * w_n  # fixed double `* *`

    return kp, kd, I_j





##
# Configuration
##

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
usd_dir_path = os.path.join(BASE_DIR, "../../../../../robot_models/")

Lus2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{usd_dir_path}/robots/lus2/usd/lus2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.0, #-0.3,
            ".*_knee_joint": 0.0, #0.3,
            ".*_ankle_pitch_joint": 0.0, #-0.18,
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0, #0.2,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0, #-0.6, 
            ".*_wrist.*": 0.0, #-0.6, 
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit_sim=380,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_yaw_joint": get_kpkd("10424")[0], #150
                ".*_hip_roll_joint": get_kpkd("10424")[0],
                ".*_hip_pitch_joint": get_kpkd("10424")[0],
                ".*_knee_joint": get_kpkd("10424")[0],
                "torso_joint": get_kpkd("10424")[0],
            },
            damping={
                ".*_hip_yaw_joint": get_kpkd("10424")[1],
                ".*_hip_roll_joint": get_kpkd("10424")[1],
                ".*_hip_pitch_joint": get_kpkd("10424")[1],
                ".*_knee_joint": get_kpkd("10424")[1],
                "torso_joint": get_kpkd("10424")[1],
            },
            armature={
                ".*_hip_.*": get_kpkd("10424")[2],
                ".*_knee_joint": get_kpkd("10424")[2],
                "torso_joint": get_kpkd("10424")[2],
            },
            friction=0.001,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*_joint"],
            effort_limit_sim=80,
            velocity_limit_sim=15.0,
            stiffness={".*_ankle.*": get_kpkd("6030")[0]},
            damping={".*_ankle.*": get_kpkd("6030")[1]},
            armature=get_kpkd("6030")[2],
            friction=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_shoulder_pitch_joint": get_kpkd("6030")[0],
                ".*_shoulder_roll_joint": get_kpkd("6030")[0],
                ".*_shoulder_yaw_joint": get_kpkd("6030")[0],
                ".*_elbow_joint": get_kpkd("6030")[0],
            },
            damping={
                ".*_shoulder_pitch_joint": get_kpkd("6030")[1],
                ".*_shoulder_roll_joint": get_kpkd("6030")[1],
                ".*_shoulder_yaw_joint": get_kpkd("6030")[1],
                ".*_elbow_joint": get_kpkd("6030")[1],
            },
            armature={
                ".*_shoulder_.*": get_kpkd("6030")[2],
                ".*_elbow_.*": get_kpkd("6030")[2],
            },
            friction=0.001,
        ),

        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_wrist_.*_joint": get_kpkd("6005")[0],
            },
            damping={
                ".*_wrist_.*_joint": get_kpkd("6005")[1],
            },
            armature=get_kpkd("6005")[2],
            friction=0.001,
        ),
    },
)

Lus2_Joint25_CFG = Lus2_CFG.copy()
Lus2_Joint25_CFG.spawn.usd_path = f"{usd_dir_path}/robots/lus2/usd/lus2_joint25.usd"

Lus2_Joint27_CFG = Lus2_CFG.copy()
Lus2_Joint27_CFG.spawn.usd_path = f"{usd_dir_path}/robots/lus2/usd/lus2_joint27.usd"

Lus2_Joint21_CFG = Lus2_CFG.copy()
Lus2_Joint21_CFG.spawn.usd_path = f"{usd_dir_path}/robots/lus2/usd/lus2_joint21.usd"
Lus2_Joint21_CFG.init_state.joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.0, #-0.3,
            ".*_knee_joint": 0.0, #0.3,
            ".*_ankle_pitch_joint": 0.0, #-0.18,
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0, #0.2,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0, #-0.6, 
        }
Lus2_Joint21_CFG.actuators.pop("wrist")
Lus2_Joint21_CFG_ImplictActuator = Lus2_Joint21_CFG.copy()

Lus2_Joint21_CFG_DelayActuator= Lus2_Joint21_CFG.copy()
Lus2_Joint21_CFG_DelayActuator.actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit_sim=300,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_yaw_joint": get_kpkd("10424")[0], #150
                ".*_hip_roll_joint": get_kpkd("10424")[0],
                ".*_hip_pitch_joint": get_kpkd("10424")[0],
                ".*_knee_joint": get_kpkd("10424")[0],
                "torso_joint": get_kpkd("10424")[0],
            },
            damping={
                ".*_hip_yaw_joint": get_kpkd("10424")[1],
                ".*_hip_roll_joint": get_kpkd("10424")[1],
                ".*_hip_pitch_joint": get_kpkd("10424")[1],
                ".*_knee_joint": get_kpkd("10424")[1],
                "torso_joint": get_kpkd("10424")[1],
            },
            armature={
                ".*_hip_.*": get_kpkd("10424")[2],
                ".*_knee_joint": get_kpkd("10424")[2],
                "torso_joint": get_kpkd("10424")[2],
            },
            friction=0.001,
            min_delay=0,
            max_delay=3,
        ),
        "feet_pitch": RemotizedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=80,
            velocity_limit_sim=15.0,
            stiffness={".*_ankle_pitch_joint": get_kpkd("6030")[0]},
            damping={".*_ankle_pitch_joint": get_kpkd("6030")[1]},
            armature=get_kpkd("6030")[2],
            min_delay=0,
            max_delay=3,
            joint_parameter_lookup=[[-0.49, 1.0, 70],[0.0, 1.0, 80],[1.1, 1.0, 60]]
        ),
        "feet_roll": RemotizedPDActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim=80,
            velocity_limit_sim=15.0,
            stiffness={".*_ankle_roll_joint": get_kpkd("6030")[0]},
            damping={".*_ankle_roll_joint": get_kpkd("6030")[1]},
            armature=get_kpkd("6030")[2],
            min_delay=0,
            max_delay=3,
            joint_parameter_lookup=[[-0.48, 1.0, 60],[0.0, 1.0, 80],[0.48, 1.0, 60]]
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_shoulder_pitch_joint": get_kpkd("6030")[0],
                ".*_shoulder_roll_joint": get_kpkd("6030")[0],
                ".*_shoulder_yaw_joint": get_kpkd("6030")[0],
                ".*_elbow_joint": get_kpkd("6030")[0],
            },
            damping={
                ".*_shoulder_pitch_joint": get_kpkd("6030")[1],
                ".*_shoulder_roll_joint": get_kpkd("6030")[1],
                ".*_shoulder_yaw_joint": get_kpkd("6030")[1],
                ".*_elbow_joint": get_kpkd("6030")[1],
            },
            armature={
                ".*_shoulder_.*": get_kpkd("6030")[2],
                ".*_elbow_.*": get_kpkd("6030")[2],
            },
            min_delay=0,
            max_delay=3,

        ),

        "wrist": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wrist_.*_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_wrist_.*_joint": get_kpkd("6005")[0],
            },
            damping={
                ".*_wrist_.*_joint": get_kpkd("6005")[1],
            },
            armature=get_kpkd("6005")[2],
            min_delay=0,
            max_delay=1,
        ),
}


Lus2_Joint21_CFG_StDelayActuator = Lus2_Joint21_CFG.copy()
Lus2_Joint21_CFG_StDelayActuator.actuators={
        "legs": StDelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit_sim=360,
            velocity_limit_sim=11.0,
            stiffness={
                ".*_hip_yaw_joint": get_kpkd("10424")[0], #150
                ".*_hip_roll_joint": get_kpkd("10424")[0],
                ".*_hip_pitch_joint": get_kpkd("10424")[0],
                ".*_knee_joint": get_kpkd("10424")[0],
                "torso_joint": get_kpkd("10424")[0],
            },
            damping={
                ".*_hip_yaw_joint": get_kpkd("10424")[1],
                ".*_hip_roll_joint": get_kpkd("10424")[1],
                ".*_hip_pitch_joint": get_kpkd("10424")[1],
                ".*_knee_joint": get_kpkd("10424")[1],
                "torso_joint": get_kpkd("10424")[1],
            },
            armature={
                ".*_hip_.*": get_kpkd("10424")[2],
                ".*_knee_joint": get_kpkd("10424")[2],
                "torso_joint": get_kpkd("10424")[2],
            },
            friction=0.01,
            effort_strength=(0.9,1.1)
        ),
        "feet": StDelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_.*"],
            effort_limit_sim=80,
            velocity_limit_sim=14.0,
            stiffness={".*_ankle_.*": get_kpkd("6030")[0]},
            damping={".*_ankle_.*": get_kpkd("6030")[1]},
            armature=get_kpkd("6030")[2],
            friction=0.01,
            effort_strength=(0.8,1.2)
        ),
        
        "arms": StDelayedPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=80,
            velocity_limit_sim=14.0,
            stiffness={
                ".*_shoulder_pitch_joint": get_kpkd("6030")[0],
                ".*_shoulder_roll_joint": get_kpkd("6030")[0],
                ".*_shoulder_yaw_joint": get_kpkd("6030")[0],
                ".*_elbow_joint": get_kpkd("6030")[0],
            },
            damping={
                ".*_shoulder_pitch_joint": get_kpkd("6030")[1],
                ".*_shoulder_roll_joint": get_kpkd("6030")[1],
                ".*_shoulder_yaw_joint": get_kpkd("6030")[1],
                ".*_elbow_joint": get_kpkd("6030")[1],
            },
            armature={
                ".*_shoulder_.*": get_kpkd("6030")[2],
                ".*_elbow_.*": get_kpkd("6030")[2],
            },
            friction=0.001,
            effort_strength=(0.9,1.1)
        )
}

Lusl1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{usd_dir_path}/robots/lusl1/usd/lusl1_joint21.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit_sim=80,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_yaw_joint": get_kpkd("6030")[0], #150
                ".*_hip_roll_joint": get_kpkd("6030")[0],
                ".*_hip_pitch_joint": get_kpkd("6030")[0],
                ".*_knee_joint": get_kpkd("6030")[0],
                "torso_joint": get_kpkd("6030")[0],
            },
            damping={
                ".*_hip_yaw_joint": get_kpkd("6030")[1],
                ".*_hip_roll_joint": get_kpkd("6030")[1],
                ".*_hip_pitch_joint": get_kpkd("6030")[1],
                ".*_knee_joint": get_kpkd("6030")[1],
                "torso_joint": get_kpkd("6030")[1],
            },
            armature={
                ".*_hip_.*": get_kpkd("6030")[2],
                ".*_knee_joint": get_kpkd("6030")[2],
                "torso_joint": get_kpkd("6030")[2],
            },
            friction=0.001,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*_joint"],
            effort_limit_sim=60,
            velocity_limit_sim=15.0,
            stiffness={".*_ankle.*": get_kpkd("6005")[0]},
            damping={".*_ankle.*": get_kpkd("6005")[1]},
            armature=get_kpkd("6005")[2],
            friction=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=80,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_shoulder_pitch_joint": get_kpkd("6030")[0],
                ".*_shoulder_roll_joint": get_kpkd("6030")[0],
                ".*_shoulder_yaw_joint": get_kpkd("6030")[0],
                ".*_elbow_joint": get_kpkd("6030")[0],
            },
            damping={
                ".*_shoulder_pitch_joint": get_kpkd("6030")[1],
                ".*_shoulder_roll_joint": get_kpkd("6030")[1],
                ".*_shoulder_yaw_joint": get_kpkd("6030")[1],
                ".*_elbow_joint": get_kpkd("6030")[1],
            },
            armature={
                ".*_shoulder_.*": get_kpkd("6030")[2],
                ".*_elbow_.*": get_kpkd("6030")[2],
            },
            friction=0.001,
        ),

    },
)

# 使用类处理不同的配置
calculator = ActionScaleCalculator(scale_factor=0.25)

Lus2_joint21_ACTION_SCALE = calculator.calculate(Lus2_Joint21_CFG)
Lus2_joint27_ACTION_SCALE = calculator.calculate(Lus2_Joint27_CFG)
Lusl1_joint21_ACTION_SCALE = calculator.calculate(Lusl1_CFG)


