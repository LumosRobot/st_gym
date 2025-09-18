import glob, os
# configurations of amp
from legged_robots.tasks.utils.wrappers.st_rl import (
    StRlAmpCfg,
    StRlAmpDataCfg,
)
############## Demo trajectory ###################
num_envs=4096
aug_traj_num=num_envs

motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/data/lus1/txt/Dance_113_04*.txt")
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
            "root_vel_x",
            "root_vel_y",
            "root_vel_z",
            "root_ang_vel_x",
            "root_ang_vel_y",
            "root_ang_vel_z",
        ]
all_joint_names = ['left_hip_roll_joint', 'right_hip_roll_joint', 'torso_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint']


# NOTE, this should follow the order of that in env when loading usd model
INIT_STATE_FIELDS = INIT_ROOT_STATE_FIELDS + [key+"_dof_pos" for key in all_joint_names] + [key+"_dof_vel" for key in all_joint_names]


############## Style  (AMP) ########################
# not need to include that for normal command (base_velocity command)
style_goal_fields=[
            "root_rot_w",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
            #"root_pos_z",
        ]
#style_goal_fields = None

style_root_fields = [
        "root_pos_z",
        "root_rot_w",
        "root_rot_x",
        "root_rot_y",
        "root_rot_z",
        "root_vel_x",
        "root_vel_y",
        "root_vel_z",
        "root_ang_vel_x",
        "root_ang_vel_y",
        "root_ang_vel_z",
        ]
style_joint_name = [
        #'left_hip_roll_joint', 'right_hip_roll_joint',  'left_hip_yaw_joint', 'right_hip_yaw_joint', 
        #'left_hip_pitch_joint', 'right_hip_pitch_joint', 
        'left_knee_joint', 'right_knee_joint', 
        'torso_joint',
        'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
        'left_elbow_joint', 'right_elbow_joint', 
        'left_wrist_yaw_joint', 'right_wrist_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint']

style_body_name = None #["left_elbow_link","right_elbow_link"] #+ ["left_ankle_link","right_ankle_link"]

style_fields = style_root_fields + \
        [key + "_dof_pos" for key in style_joint_name] + \
        [key + "_dof_vel" for key in style_joint_name] + \
        ([k1 + "_" + k2 for k1 in style_body_name for k2 in ["x", "y", "z"]] if style_body_name is not None else [])

style_reward_coef = 1.0

##############  Expressive (Mimic) ########################

expressive_link_name =  None #["left_elbow_link","right_elbow_link"]

expressive_joint_name = ["torso_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint","left_shoulder_roll_joint","right_shoulder_roll_joint",  "left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint","right_wrist_yaw_joint"] #"left_hip_pitch_joint", "right_hip_pitch_joint"] # "left_hip_yaw_joint", "right_hip_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "left_knee_joint","right_knee_joint"]


# not need to include that for normal command (base_velocity command)
expressive_goal_fields= [key+"_dof_pos" for key in expressive_joint_name]



track_expressive_joint_pos_weight = 1.0 #4
track_expressive_link_pos_weight =  1.47

###################### amp discriminator #################
amp_discriminator = StRlAmpCfg(
        input_dim = int(len(style_fields) * amp_obs_frame_num),  # torch.cat([amp_obs, next_amp_obs],dim=-1)
        style_reward_coef=style_reward_coef, #0.18
        discriminator_grad_pen = 10.0,
        hidden_dims=[512, 512, 512],
        task_reward_lerp=0.5, #0.7
)

###################### amp data #################
num_steps_per_env = 24
amp_data = StRlAmpDataCfg(
        motion_files=motion_files,
        init_state_fields=INIT_STATE_FIELDS,
        style_goal_fields=style_goal_fields, # as input for the policy
        style_fields=style_fields, # as for style rewards
        expressive_goal_fields = expressive_goal_fields, # only as input for the policy
        expressive_joint_name = expressive_joint_name, # for tracking rewards
        expressive_link_name = expressive_link_name, # for tracking rewards
        time_between_frames=0.02,  # time between two frames of state and next_state
        shuffle=False,
        random_start=random_start,
        amp_obs_frame_num=amp_obs_frame_num, #-1+1,
        discriminator_loss_coef=1.0, #1.0
        mimic_loss_coef=0.0, # 
        amp_obs_dim=int(len(style_fields) * amp_obs_frame_num), 
        augment_trajectory_num=aug_traj_num,
        augment_frame_num=num_steps_per_env,
        frame_begin=10,
        frame_end=None,
)

##################### Algorithm  and Plocy ###################
algorithm_name = "APPO"  # "PPO" #"TPPO"
policy_name = "AmpActorCritic"  # "ActorCritic" #"ActorCriticRecurrent"
runner_name = "AmpPolicyRunner"  # AmpPolicyRunner
