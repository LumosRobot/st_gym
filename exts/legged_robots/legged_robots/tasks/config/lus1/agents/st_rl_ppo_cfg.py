"""Proximal Policy Optimization (PPO) Configuration for Kepler Lus1 Robot.

This module defines the configuration for training Lus1 robot using PPO algorithm
in rough terrain environments. It includes settings for the policy network,
value function, and training parameters.

Note:
    This configuration extends the base ST-RL PPO implementation with
    specific parameters tuned for Lus1 robot locomotion tasks.
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
)

# Import AMP data configurations
from legged_robots.tasks.locomotion.velocity.config.lus1.amp_data_cfg import *

@configclass
class Lus1RoughPPORunnerCfg(StRlOnPolicyRunnerCfg):
    """Configuration class for Lus1 robot PPO training in rough terrain.
    
    This class inherits from StRlOnPolicyRunnerCfg and defines specific
    parameters for training Lus1 robot using PPO algorithm.
    
    Attributes:
        num_steps_per_env (int): Number of steps to run per environment
        max_iterations (int): Maximum number of training iterations
        save_interval (int): Frequency of saving model checkpoints
        experiment_name (str): Name of the experiment for logging
        empirical_normalization (bool): Whether to use empirical normalization
        policy (StRlPpoActorCriticCfg): Actor-critic network configuration
        algorithm (StRlPpoAlgorithmCfg): PPO algorithm parameters
    """
    
    num_steps_per_env = num_steps_per_env  # Steps per environment
    max_iterations = 3000                   # Maximum training iterations
    save_interval = 100                      # Checkpoint saving frequency
    experiment_name = "lus1_rough"            # Experiment identifier
    empirical_normalization = True          # Use empirical normalization

    # Actor-critic network configuration
    policy = StRlPpoActorCriticCfg(
        init_noise_std=1.0,                 # Initial exploration noise
        actor_hidden_dims=[512, 256, 128],  # Actor network architecture
        critic_hidden_dims=[512, 256, 128], # Critic network architecture
        activation="elu",                   # Activation function
        rnn_type="gru",                     # Recurrent neural network type
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

    # storage configs
    storage = OrderedDict(
        amp_obs_shape=(len(style_fields) * amp_obs_frame_num,),
    )

    # log configs
    log_interval = 10
    #logger="neptune"
    logger="tensorboard"
    neptune_project="lus1-rl"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = algorithm_name  # "PPO" #"TPPO"
        self.policy.class_name = policy_name  # "ActorCritic" #"ActorCriticRecurrent"
        self.runner_class_name = runner_name  # AmpPolicyRunner
        if self.algorithm.class_name=="APPO":
            setattr(self.policy, "amp_discriminator", amp_discriminator)
            setattr(self.algorithm, "amp_data", amp_data)


@configclass
class Lus1FlatPPORunnerCfg(Lus1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 6000
        self.experiment_name = "lus1_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
        self.algorithm.class_name = algorithm_name  # "PPO" #"TPPO"
        self.policy.class_name = policy_name  # "ActorCritic" #"ActorCriticRecurrent"
        self.runner_class_name = runner_name  # AmpPolicyRunner
        if self.algorithm.class_name=="APPO":
            setattr(self.policy, "amp_discriminator", amp_discriminator)
            setattr(self.algorithm, "amp_data", amp_data)
