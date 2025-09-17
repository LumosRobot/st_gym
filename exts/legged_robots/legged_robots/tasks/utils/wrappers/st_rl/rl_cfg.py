# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

@configclass
class StRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    rnn_type: str = MISSING
    """The network type for the actor and critic networks."""

    teacher_ac_path: str = None

    num_actor_obs: int = None
    num_critic_obs: int = None
    num_actions: int = None


    encoder_component_names: list = None
    obs_segments = None
    critic_encoder_component_names: list = None
    critic_obs_segments = None
    amp_discriminator = None
    mu_activation = None


@configclass
class StRlAmpCfg:
    """Configuration for the AMP networks."""

    input_dim: int = MISSING # torch.cat([amp_obs, next_amp_obs],dim=-1)
    style_reward_coef: float = MISSING
    discriminator_grad_pen: float = MISSING
    hidden_dims: list[int] = MISSING
    task_reward_lerp: float = MISSING
    amp_obs_dim: int=MISSING




@configclass
class StRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    mimic_loss_coef: float =0.0
    discriminator_loss_coef: float=0.0
    amp_obs_dim: int = 2


@configclass
class StRlTppoAlgorithmCfg:
    """Configuration for Teacher-PPO or distillation-based RL algorithm."""

    ppo: StRlPpoAlgorithmCfg = None

    teacher_policy: StRlPpoActorCriticCfg = MISSING

    label_action_with_critic_obs: bool = True
    teacher_act_prob: str = "exp"  # could also be a float or callable
    update_times_scale: int = 100
    using_ppo: bool = True  # if False, behaves like DAgger

    distillation_loss_coef: float | str = 1.0
    distill_target: str = "real"
    distill_latent_coef: float = 1.0
    distill_latent_target: str = "real"
    distill_latent_obs_component_mapping: dict | None = None

    buffer_dilation_ratio: float = 1.0
    lr_scheduler_class_name: str | None = None
    lr_scheduler: dict = {}

    hidden_state_resample_prob: float = 0.0
    action_labels_from_sample: bool = False



@configclass
class StRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: StRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: StRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
