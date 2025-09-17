"""Proximal Policy Optimization (PPO) Configuration for Kepler Lus2 Robot.

This module defines the configuration for training Lus2 robot using PPO algorithm
in rough terrain environments. It includes settings for the policy network,
value function, and training parameters.

Note:
    This configuration extends the base ST-RL PPO implementation with
    specific parameters tuned for Lus2 robot locomotion tasks.
"""

# Standard library imports
from collections import OrderedDict

# Isaac Lab utilities
from isaaclab.utils import configclass

# ST-RL wrapper configurations
from legged_robots.tasks.utils.wrappers.st_rl import (
    StRlOnPolicyRunnerCfg,
    StRlPpoActorCriticCfg,
    StRlPpoAlgorithmCfg,
    StRlTppoAlgorithmCfg,
)

# Import AMP data configurations
from legged_robots.tasks.config.lus2.amp_mimic_cfg import *

@configclass
class Lus2RoughPPORunnerCfg(StRlOnPolicyRunnerCfg):
    """Configuration class for Lus2 robot PPO training in rough terrain.
    
    This class inherits from StRlOnPolicyRunnerCfg and defines specific
    parameters for training Lus2 robot using PPO algorithm.
    
    Attributes:
        num_steps_per_env (int): Number of steps to run per environment
        max_iterations (int): Maximum number of training iterations
        save_interval (int): Frequency of saving model checkpoints
        experiment_name (str): Name of the experiment for logging
        empirical_normalization (bool): Whether to use empirical normalization
        policy (StRlPpoActorCriticCfg): Actor-critic network configuration
        algorithm (StRlPpoAlgorithmCfg): PPO algorithm parameters
    """
    
    num_steps_per_env = 24  # Steps per environment
    max_iterations = 1000                   # Maximum training iterations
    save_interval = 100                      # Checkpoint saving frequency
    experiment_name = "lus2_rough"            # Experiment identifier
    empirical_normalization = True          # Use empirical normalization

    # Actor-critic network configuration
    policy = StRlPpoActorCriticCfg(
        init_noise_std=init_noise_std,                 # Initial exploration noise
        actor_hidden_dims=[512, 256, 128],  # Actor network architecture
        critic_hidden_dims=[512, 256, 128], # Critic network architecture
        activation="elu",                   # Activation function
        rnn_type="lstm",                     # Recurrent neural network type
        #mu_activation="tanh",
    )

    # PPO algorithm configuration
    algorithm = StRlPpoAlgorithmCfg(
        value_loss_coef=1.0,               # Value loss coefficient
        use_clipped_value_loss=True,       # Use PPO clipped value loss
        clip_param=0.2,                    # PPO clipping parameter
        entropy_coef=0.01,                 # Entropy coefficient
        num_learning_epochs=5,             # Number of learning epochs
        num_mini_batches=4,                # Number of mini-batches
        learning_rate=1.0e-3,              # Learning rate
        schedule="adaptive",               # Learning rate schedule
        gamma=0.99,                        # Discount factor
        lam=0.95,                          # GAE lambda
        desired_kl=0.01,                   # Desired KL divergence
        max_grad_norm=1.0,                 # Maximum gradient norm
        empirical_normalization=empirical_normalization,
    )
    if "TPPO" in algorithm_name:
        algorithm = StRlTppoAlgorithmCfg(
                ppo = algorithm, # running device will be handled
                teacher_policy = teacher_policy,
                label_action_with_critic_obs= True, # else, use actor obs
                teacher_act_prob= "exp", # a number or a callable to (0 ~ 1) to the selection of act using teacher policy
                update_times_scale= 100, # a rough estimation of how many times the update will be called
                using_ppo= True, # If False, compute_losses will skip ppo loss computation and returns to DAGGR
                distillation_loss_coef= 1., # can also be string to select a prob function to scale the distillation loss
                distill_target= "real",
                distill_latent_coef= 1.,
                distill_latent_target= "real",
                distill_latent_obs_component_mapping = distill_latent_obs_component_mapping,
                buffer_dilation_ratio= 1.,
                lr_scheduler_class_name= None,
                lr_scheduler= dict(),
                hidden_state_resample_prob= 0.0, # if > 0, Some hidden state in the minibatch will be resampled
                action_labels_from_sample= False, # if True, the action labels from teacher policy will be from policy.act instead of policy.act_inference
                )

    # storage configs
    storage = OrderedDict(
        amp_obs_shape=(len(style_fields) * amp_obs_frame_num,),
    )

    # log configs
    log_interval = 10
    logger="neptune"
    #logger="tensorboard"
    neptune_project="lus2-rl"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = algorithm_name  # "PPO" #"TPPO"
        self.policy.class_name = policy_name  # "ActorCritic" #"ActorCriticRecurrent"
        self.runner_class_name = runner_name  # AmpPolicyRunner
        if "A" in self.algorithm.class_name:
            setattr(self.policy, "amp_discriminator", amp_discriminator)
            self.algorithm.bc_loss_coef = bc_loss_coef
            self.algorithm.discriminator_loss_coef = discriminator_loss_coef
            self.algorithm.amp_obs_dim=amp_obs_dim

        if "Encoder" in self.policy.class_name:
            self.policy.encoder_component_names = encoder_component_names
            self.policy.obs_segments=obs_segments
            self.policy.critic_encoder_component_names = critic_encoder_component_names
            self.policy.critic_obs_segments = critic_obs_segments


@configclass
class Lus2FlatPPORunnerCfg(Lus2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.save_interval = 500     # Checkpoint saving frequency
        self.max_iterations = 20000
        self.experiment_name = "lus2_flat"
        self.algorithm.class_name = algorithm_name  # "PPO" #"TPPO"
        self.policy.class_name = policy_name  # "ActorCritic" #"ActorCriticRecurrent"
        self.runner_class_name = runner_name  # AmpPolicyRunner
        if "A" in self.algorithm.class_name:
            setattr(self.policy, "amp_discriminator", amp_discriminator)
            self.algorithm.bc_loss_coef = bc_loss_coef
            self.algorithm.discriminator_loss_coef = discriminator_loss_coef
            self.algorithm.amp_obs_dim=amp_obs_dim
