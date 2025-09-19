import glob, os
from collections import OrderedDict

# configurations of amp
from legged_robots.tasks.utils.wrappers.st_rl import (
    StRlAmpCfg,
    StRlPpoActorCriticCfg,
)

#from legged_robots.data_manager.motion_loader import RefMotionCfg
from refmotion_manager.motion_loader import RefMotionCfg

############## Demo trajectory ###################
num_envs = 1000 #4096
using_21_joint = True

# 27 joints
if not using_21_joint:
    motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/CMU_CMU_13_13*_fps*.pkl")
else:
    # 21 joints
    motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/CMU_CMU_13_13*.pkl")


random_start=True
amp_obs_frame_num = 2 # minimal is 1, no history amp obs

############## Init states (carefully)  ###################
# constants
INIT_ROOT_STATE_FIELDS = [
            "root_pos_x",
            "root_pos_y",
            "root_pos_z",
            "root_rot_w",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
            "root_vel_x_b",
            "root_vel_y_b",
            "root_vel_z_b",
            "root_ang_vel_x_b",
            "root_ang_vel_y_b",
            "root_ang_vel_z_b",
        ]

# 27 joints
lus2_27joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
'left_wrist_roll_joint', 'right_wrist_roll_joint'
]

# 21 joints
lus2_21joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint'
]

all_joint_names = lus2_21joint_names if using_21_joint else lus2_27joint_names

# NOTE, this should follow the order of that in env when loading usd model
INIT_STATE_FIELDS = INIT_ROOT_STATE_FIELDS + [key+"_dof_pos" for key in all_joint_names] + [key+"_dof_vel" for key in all_joint_names]


############## Style  (AMP) ########################
using_amp=True
# not need to include that for normal command (base_velocity command)
style_root_fields = [
        #"root_pos_z",
        "root_rot_w",
        "root_rot_x",
        "root_rot_y",
        "root_rot_z",
        "root_vel_x_b",
        "root_vel_y_b",
        "root_vel_z_b",
        "root_ang_vel_x_b",
        "root_ang_vel_y_b",
        "root_ang_vel_z_b",
        ]

# 27 joints
style_joint_name = [
        'left_hip_roll_joint', 'right_hip_roll_joint',  'left_hip_yaw_joint', 'right_hip_yaw_joint', 
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 
        'left_knee_joint', 'right_knee_joint', 
        'torso_joint',
        'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
        'left_elbow_joint', 'right_elbow_joint'] + [] if using_21_joint else ['left_wrist_yaw_joint', 'right_wrist_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']


style_body_name = ["left_elbow_link", "right_elbow_link","left_hip_pitch_link","right_hip_pitch_link","left_shoulder_roll_link","right_shoulder_roll_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link","left_wrist_roll_link","right_wrist_roll_link"] + ["left_elbow_link", "left_hip_pitch_link", "right_hip_pitch_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link"]
style_body_name = None

style_fields = style_root_fields + \
        ([key + "_dof_pos" for key in style_joint_name] if style_joint_name is not None else []) + \
        ([key + "_dof_vel" for key in style_joint_name] if style_joint_name is not None else []) + \
        ([k1 + k2 for k1 in style_body_name for k2 in ["_pos_x_b", "_pos_y_b", "_pos_z_b"]] if style_body_name is not None else [])


# carefully to change this
style_goal_fields=[
            "root_rot_w",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
        ]

style_goal_fields = None
track_style_goal_weight =  1.47 # the goal of syle
style_reward_coef = 1.0

##############  Expressive (Mimic) ########################
using_mimic = False

#link_name =  ["right_knee_link", "left_elbow_link"]
expressive_link_name = ["left_elbow_link", "right_elbow_link","left_hip_pitch_link","right_hip_pitch_link","left_shoulder_roll_link","right_shoulder_roll_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link","left_wrist_roll_link","right_wrist_roll_link"]
expressive_link_name =  ["left_hip_pitch_link", "right_hip_pitch_link","left_ankle_roll_link","right_ankle_roll_link", "left_knee_link", "right_knee_link"]
expressive_link_name = None

# 27 joints robot
upper_joint_name = ["torso_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint","left_shoulder_roll_joint","right_shoulder_roll_joint",  "left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint", "right_elbow_joint"] + [] if using_21_joint else ["left_wrist_yaw_joint","right_wrist_yaw_joint", 'left_wrist_roll_joint', 'right_wrist_roll_joint']

lower_joint_name = ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "left_knee_joint","right_knee_joint"]
feet_joint_name = ["left_ankle_pitch_joint", "right_ankle_pitch_joint","left_ankle_roll_joint","right_ankle_roll_joint"]
joint_name = upper_joint_name + lower_joint_name + feet_joint_name

# not need to include that for normal command (base_velocity command)
expressive_goal_fields = [key+"_dof_pos" for key in joint_name] if joint_name is not None else None
expressive_goal_fields += [key+"_dof_vel" for key in joint_name] if joint_name is not None else []

expressive_goal_fields += [key1+key2 for key1 in expressive_link_name for key2 in ["_pos_x_b","_pos_y_b","_pos_z_b"]] if expressive_link_name is not None else []
expressive_goal_fields += [key1+key2 for key1 in expressive_link_name for key2 in ["_vel_x_b","_vel_y_b","_vel_z_b"]] if expressive_link_name is not None else []

feet_parallel_v1_weight = 1.0
feet_stumble_weight = -500.0

track_lin_vel_xy_weight = 2.5
track_ang_vel_z_weight = 2.0
track_upper_joint_pos_weight = 20.4 #4
track_upper_joint_vel_weight = 3.5 #4

track_lower_joint_pos_weight = 15.0 #4
track_lower_joint_vel_weight = 2.2 #4

track_feet_joint_pos_weight = 2.0 #4
track_feet_joint_vel_weight = 1.0 # #4

track_link_pos_weight =  0
track_link_vel_weight =  0

track_root_pos_weight =  2.0
track_root_rotation_weight = 2.0
track_root_quat_weight = 0.0
track_root_lin_vel_weight =  1 #1.0
track_root_ang_vel_weight =  1 #1.0

###################### amp discriminator #################
amp_discriminator = StRlAmpCfg(
        input_dim = int(len(style_fields) * amp_obs_frame_num),  # torch.cat([amp_obs, next_amp_obs],dim=-1)
        style_reward_coef=style_reward_coef, #0.18
        discriminator_grad_pen = 10.0,
        hidden_dims=[512, 512, 512],
        task_reward_lerp=0.0, #0.7
)

# for appo
bc_loss_coef=0.001 # behavior cloning
amp_obs_dim=int(len(style_fields) * amp_obs_frame_num) 
episode_length_s = 2
init_noise_std = 1.2

###################### amp data #################
specify_init_values = {key+"_dof_pos": 0.0 for key in joint_name} # same with real robot stand state posture
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


ref_motion_cfg = RefMotionCfg(
        motion_files=motion_files,
        init_state_fields=INIT_STATE_FIELDS,
        style_goal_fields=style_goal_fields, # as input for the policy
        style_fields=style_fields, # as for style rewards
        expressive_goal_fields = expressive_goal_fields, # only as input for the policy
        expressive_joint_name = joint_name, # for tracking rewards
        expressive_link_name = expressive_link_name, # for tracking rewards
        time_between_frames=0.02,  # time between two frames of state and next_state
        shuffle=False, #shuffle different trajectories
        random_start=random_start,
        amp_obs_frame_num=amp_obs_frame_num, #-1+1,
        ref_length_s=episode_length_s, # 20 s
        trajectory_num=num_envs,
        frame_begin=None,
        frame_end= None,
        specify_init_values = None #specify_init_values # None
)

##################### Algorithm  and Policy ###################
algorithm_name = "APPO" #"ATPPO"  # "PPO" #"TPPO", "APPO"
policy_name = "AmpActorCriticRecurrent" #"AmpActorCritic" #"AmpActorCriticRecurrent"  # "ActorCritic" #"ActorCriticRecurrent"
#policy_name = "EncoderAmpActorCriticRecurrent"  # "ActorCritic" #"ActorCriticRecurrent"
runner_name = "AmpPolicyRunner"  # AmpPolicyRunner
training_teacher = False

# NOTE: setting the follow obs_segments accoding to actual obs terms
if training_teacher:
    encoder_component_names = ["privileged_obs"]
    obs_segments = OrderedDict({"obs":(103),"privileged_obs":(91)})
    critic_encoder_component_names = ["privileged_obs"]
    critic_obs_segments = OrderedDict({"obs":(103),"privileged_obs":(91)})
else:
    if "Encoder" in policy_name:
        # actor for student
        encoder_component_names= ["joint_obs"]
        obs_segments = OrderedDict({"obs":(93),"joint_obs":(21*3)})
        # critic for student, it use prvileged info, so same with teacher obs
        critic_encoder_component_names = ["privileged_obs"]
        critic_obs_segments = OrderedDict({"obs":(103),"privileged_obs":(91)})
    # distill
distill_latent_obs_component_mapping = {"joint_obs": "privileged_obs"}


##################### DISTILL ##########################
teacher_ac_path = os.getenv("HOME")+"/workspace/lumos_ws/st_gym/logs/st_rl/lus2_flat/2025-07-19_16-45-53/model_19000.pt"
teacher_policy = StRlPpoActorCriticCfg(
        teacher_ac_path= teacher_ac_path,
        num_actor_obs=103+91,
        num_critic_obs=103+91,
        num_actions=21,
        class_name= "EncoderAmpActorCriticRecurrent",
        init_noise_std=1.0,                 # Initial exploration noise
        actor_hidden_dims=[512, 256, 128],  # Actor network architecture
        critic_hidden_dims=[512, 256, 128], # Critic network architecture
        activation="elu",                   # Activation function
        rnn_type="lstm",                     # Recurrent neural network type
        encoder_component_names = ["privileged_obs"],
        obs_segments= OrderedDict({"obs":(103),"privileged_obs":(91)}),
        critic_encoder_component_names = ["privileged_obs"],
        critic_obs_segments= OrderedDict({"obs":(103),"privileged_obs":(91)}),
        amp_discriminator = amp_discriminator
        )



if "A" in algorithm_name:
    discriminator_loss_coef=1.0
else:
    discriminator_loss_coef=0.0

#if algorithm_name == "TPPO" or algorithm_name == "ATPPO":
#    training_type = "distill"
#else:
#    training_type="rl"

