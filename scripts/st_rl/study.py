import os
import json
from collections import OrderedDict

import sys
import optuna
from optuna.trial import TrialState
import neptune
import neptune.integrations.optuna as optuna_utils
import yaml
import subprocess
import time

# optuna
def objective(trial):
    print("start a training process")
    # save tran_cfg, env_cfg, and env

    # with open(os.path.join(log_dir, "config.json"), "r") as f:
    # cfg_dict = json.load(f, object_pairs_hook= OrderedDict) # read
    # cfg_dict = {"rewards":{}, "domain_rand":{}, "commands": {}, "env":{}, "control":{}}

    # 1) rewards, all rewards are here
    # cfg_dict['rewards']['scales']['termination'] = trial.suggest_float("rewards/scales/termination", -3, 0)
    # cfg_dict['rewards']['scales']['tracking_lin_vel'] = trial.suggest_float("rewards/scales/tracking_lin_vel", 2, 8)
    # cfg_dict['rewards']['scales']['tracking_ang_vel'] = trial.suggest_float("rewards/scales/tracking_ang_vel", 2, 8)
    # cfg_dict['rewards']['scales']['lin_vel_z'] = trial.suggest_float("rewards/scales/lin_vel_z", -3.0, 0.0)
    # cfg_dict['rewards']['scales']['ang_vel_xy'] = trial.suggest_float("rewards/scales/ang_vel_xy", -0.5, 0)
    # cfg_dict['rewards']['scales']['orientation'] = trial.suggest_float("rewards/scales/orientation", -3, -1)
    # cfg_dict['rewards']['scales']['torques'] = trial.suggest_float("rewards/scales/torques", -4e-3, -1e-5)
    # cfg_dict['rewards']['scales']['dof_pos_limits'] = trial.suggest_float("rewards/scales/dof_pos_limits", -15.0, -5.0)
    # cfg_dict['rewards']['scales']['dof_vel_limits'] = trial.suggest_float("rewards/scales/dof_vel_limits", -5e-1, -1e-3)
    # cfg_dict['rewards']['scales']['dof_torque_limits'] = trial.suggest_float("rewards/scales/dof_torque_limits", -5e-1, -1e-3)
    # cfg_dict['rewards']['scales']['dof_vel'] = trial.suggest_float("rewards/scales/dof_vel", -1e-3, -1e-7)
    # cfg_dict['rewards']['scales']['dof_acc'] = trial.suggest_float("rewards/scales/dof_acc", -1e-3, -1e-7)
    ##cfg_dict['rewards']['scales']['base_height'] = trial.suggest_float("rewards/scales/base_height", -1e-3, -1e-7)
    # cfg_dict['rewards']['scales']['stumble'] = trial.suggest_float("rewards/scales/stumble", -0.1, 0.0)
    # cfg_dict['rewards']['scales']['action_rate'] = trial.suggest_float("rewards/scales/action_rate", -0.05, 0)
    # cfg_dict['rewards']['scales']['stand_still'] = trial.suggest_float("rewards/scales/stand_still", -1.0, 0)
    # cfg_dict['rewards']['scales']['feet_air_time'] = trial.suggest_float("rewards/scales/feet_air_time",0.0, 10.0)

    # cfg_dict['rewards']['scales']['lin_vel_l2norm'] = trial.suggest_float("rewards/scales/lin_vel_l2norm",-1.0, 0.0)
    # cfg_dict['rewards']['scales']['legs_energy_substeps'] = trial.suggest_float("rewards/scales/legs_energy_substeps",-0.01, 0.0)
    # cfg_dict['rewards']['scales']['legs_energy'] = trial.suggest_float("rewards/scales/legs_energy",-0.01, 0.0)
    # cfg_dict['rewards']['scales']['alive'] = trial.suggest_float("rewards/scales/alive",0.0, 1.0)
    # cfg_dict['rewards']['scales']['exceed_dof_pos_limits'] = trial.suggest_float("rewards/scales/exceed_dof_pos_limits",-0.001, 0.0)
    # cfg_dict['rewards']['scales']['exceed_torque_limits_l1norm'] = trial.suggest_float("rewards/scales/exceed_torque_limits_l1norm",-0.01, 0.0)

    ## walk these way # 11
    # cfg_dict['rewards']['scales']['feet_clearance_cmd_linear'] = trial.suggest_float("rewards/scales/feet_clearance_cmd_linear", -40.0,0.0)
    # cfg_dict['rewards']['scales']['feet_slip'] = trial.suggest_float("rewards/scales/feet_slip", -0.4, 0.0)
    # cfg_dict['rewards']['scales']['action_smoothness_1'] = trial.suggest_float("rewards/scales/action_smoothness_1", -0.4, 0.0)
    # cfg_dict['rewards']['scales']['action_smoothness_2'] = trial.suggest_float("rewards/scales/action_smoothness_2", -0.4, 0.0)
    # cfg_dict['rewards']['scales']['jump'] = trial.suggest_float("rewards/scales/jump",  -10, 10)
    # cfg_dict['rewards']['scales']['raibert_heuristic'] = trial.suggest_float("rewards/scales/raibert_heuristic", -25, 0)
    # cfg_dict['rewards']['scales']['feet_impact_vel'] = trial.suggest_float("rewards/scales/feet_impact_vel", -1.0, 0.0)
    # cfg_dict['rewards']['scales']['orientation_control'] = trial.suggest_float("rewards/scales/orientation_control", -10, 0)
    # cfg_dict['rewards']['scales']['tracking_contacts_shaped_vel'] = trial.suggest_categorical("rewards/scales/tracking_contacts_shaped_vel", [10,5,2,1,0.5, 0])
    # cfg_dict['rewards']['scales']['tracking_contacts_shaped_force'] = trial.suggest_categorical("rewards/scales/tracking_contacts_shaped_force", [10,5,2,1,0.5, 0])
    # cfg_dict['rewards']['scales']['collision'] = trial.suggest_float("rewards/scales/collision", -10, 0)

    ## parkour
    # cfg_dict['rewards']['scales']['tracking_lin_vel'] = trial.suggest_float("rewards/scales/tracking_lin_vel", 0.5, 3)
    # cfg_dict['rewards']['scales']['tracking_ang_vel'] = trial.suggest_float("rewards/scales/tracking_ang_vel", 0.5, 3)

    # cfg_dict['rewards']['scales']['energy_substeps'] = trial.suggest_categorical("rewards/scales/energy_substeps", [-3e-6,-2e-6,-1e-6,-1e-7,0])
    # cfg_dict['rewards']['scales']['torques'] = trial.suggest_categorical("rewards/scales/torques", [-1e-5,-1e-6,-1e-7,-1e-8,0])
    # cfg_dict['rewards']['scales']['stand_still'] = trial.suggest_float("rewards/scales/stand_still", -2, 0)
    # cfg_dict['rewards']['scales']['dof_error_named'] = trial.suggest_float("rewards/scales/dof_error_named", -3,0)
    # cfg_dict['rewards']['scales']['dof_error'] = trial.suggest_float("rewards/scales/dof_error", -1,0)

    # cfg_dict['rewards']['scales']['collision'] = trial.suggest_float("rewards/scales/collision", -5, 0)
    # cfg_dict['rewards']['scales']['lazy_stop'] = trial.suggest_float("rewards/scales/lazy_stop", -6,-1)
    # cfg_dict['rewards']['scales']['exceed_dof_pos_limits'] = trial.suggest_float("rewards/scales/exceed_dof_pos_limits",-1, 0.0)
    # cfg_dict['rewards']['scales']['exceed_torque_limits_l1norm'] = trial.suggest_float("rewards/scales/exceed_torque_limits_l1norm",-1, 0.0)
    # cfg_dict['rewards']['scales']['penetrate_depth'] = trial.suggest_float("rewards/scales/penetrate_depth",-2, 0.0)

    # 2) domain_rand
    # cfg_dict['domain_rand']['push_robots'] = trial.suggest_categorical("domain_rand/push_robots", [True, False])
    # cfg_dict['domain_rand']['randomize_com_displacement'] = trial.suggest_categorical("domain_rand/randomize_com_displacement", [True, False])
    # cfg_dict['domain_rand']['randomize_friction'] = trial.suggest_categorical("domain_rand/randomize_friction", [True, False])
    # cfg_dict['domain_rand']['randomize_gravity'] = trial.suggest_categorical("domain_rand/randomize_gravity", [True, False])
    # cfg_dict['domain_rand']['randomize_ground_friction'] = trial.suggest_categorical("domain_rand/randomize_ground_friction", [True, False])
    # cfg_dict['domain_rand']['randomize_motor_offset'] = trial.suggest_categorical("domain_rand/randomize_motor_offset", [True, False])
    # cfg_dict['domain_rand']['randomize_motor_strength'] = trial.suggest_categorical("domain_rand/randomize_motor_strength", [True, False])
    # cfg_dict['domain_rand']['randomize_restitution'] = trial.suggest_categorical("domain_rand/randomize_restitution", [True, False])

    # 3) observations components
    # cfg_dict['env']={}
    # s1, s2, s3, s4 = ["proprioception"], ["proprioception", "robot_config"], ["proprioception", "robot_config", "motor_config"], ["proprioception", "robot_config", "motor_config", "base_pose"]
    # cfg_dict['env']['obs_components'] = trial.suggest_categorical("env/obs_components", ['s1','s2','s3','s4'])
    # cfg_dict['env']['privileged_obs_components'] = trial.suggest_categorical("env/privileged_obs_components", ['s1','s2','s3','s4'])

    # 4) commands
    # cfg_dict['commands']['curriculum'] = trial.suggest_categorical("commands/curriculum", [True, False])

    # 5) control
    # cfg_dict['control']['action_scale'] = trial.suggest_categorical("control/action_scale", [0.2, 0.3, 0.4, 0.6, 0.8])
    # cfg_dict['control']['stiffness']={}
    # cfg_dict['control']['stiffness']['joint1'] = trial.suggest_categorical("control/stiffness/joint1", [20,25,30,35,40])
    # cfg_dict['control']['stiffness']['joint2'] = trial.suggest_categorical("control/stiffness/joint2", [20,25,30,35,40])
    # cfg_dict['control']['stiffness']['joint3'] = trial.suggest_categorical("control/stiffness/joint3", [20,25,30,35,40])
    # cfg_dict['control']['damping']={}
    # cfg_dict['control']['damping']['joint1'] = trial.suggest_categorical("control/damping/joint1", [0.1,0.3,0.6,1.0,1.3])
    # cfg_dict['control']['damping']['joint2'] = trial.suggest_categorical("control/damping/joint2", [0.1,0.3,0.6,1.0,1.3])
    # cfg_dict['control']['damping']['joint3'] = trial.suggest_categorical("control/damping/joint3", [0.1,0.3,0.6,1.0,1.3])
    
    # configurations of amp
    cfg_dict = {"agent": {}, "env": {}}
    cfg_dict["agent"]["logger"]="tensorboard"
    cfg_dict["agent"]["max_iterations"]=2000

    cfg_dict["env"]["rewards"] = {}

    cfg_dict["env"]["rewards"]["action_rate_l2"]={}
    cfg_dict["env"]["rewards"]["action_rate_l2"]["weight"] = trial.suggest_float(
        "env/rewards/action_rate_l2/weight", -1, -1e-2)

    cfg_dict["env"]["rewards"]["track_upper_joint_pos_exp"]={}
    cfg_dict["env"]["rewards"]["track_upper_joint_pos_exp"]["weight"] = trial.suggest_float(
        "env/rewards/track_upper_joint_pos_exp/weight", 0.5, 10.0
    )

    cfg_dict["env"]["rewards"]["track_upper_joint_pos_exp"]["params"] = {}
    cfg_dict["env"]["rewards"]["track_upper_joint_pos_exp"]["params"]["std"] = trial.suggest_float(
        "env/rewards/track_upper_joint_pos_exp/params/std", 0.1,1.0
    )


    cfg_dict["env"]["rewards"]["track_lower_joint_pos_exp"]={}
    cfg_dict["env"]["rewards"]["track_lower_joint_pos_exp"]["weight"] = trial.suggest_float(
        "env/rewards/track_lower_joint_pos_exp/weight", 0.5, 10.0
    )
    cfg_dict["env"]["rewards"]["track_lower_joint_pos_exp"]["params"] = {}
    cfg_dict["env"]["rewards"]["track_lower_joint_pos_exp"]["params"]["std"] = trial.suggest_float(
        "env/rewards/track_lower_joint_pos_exp/params/std", 0.1,1.0
    )

    cfg_dict["env"]["rewards"]["dof_pos_limits"] = {}
    cfg_dict["env"]["rewards"]["dof_pos_limits"]["weight"] = trial.suggest_float(
        "env/rewards/dof_pos_limits/weight", -10,-0.5
    )

    # write config to file
    log_dir = os.path.join("./scripts/st_rl/conf/")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config1.yaml"), "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

    ##----------------------BEGIN of  Neptune ------------------##
    # save log to neptune
    run_trial_level = neptune.init_run(
        project="suntao/ambot-w1-rl",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        mode="offline",
    )

    # log sweep id to trial-level run
    run_trial_level["sys/tags"].add("trial-level")
    # save log to neptune
    for key, value in cfg_dict.items():
        run_trial_level[key] = value

    ##----------------------END of  Neptune ------------------##

    # delete the results
    result_path = "./scripts/st_rl/conf/results.json"
    if os.path.exists(result_path):
        os.remove(result_path)
    
    # training process
    try:
        result = subprocess.run(
            ["/usr/bin/bash", "./scripts/st_rl/run_train.sh", "-d", DEVICE, "-r", "-l","2025-07-05_23-49-53"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("Script output:", result.stdout)
        print("Script errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error: {e}")
        exit()

    # wait for the results file to be created
    # while not os.path.exists(result_path):
    #time.sleep(1)

    # read the results
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            results = json.load(f)
        metrics_velrmsd = results["metrics_velrmsd"]
        metrics_CoT = results["metrics_CoT"]
    else:
        metrics_velrmsd = 100
        metrics_CoT = 100
    # log results to neptune
    run_trial_level["metrics/lin_vel_track"] = metrics_velrmsd
    run_trial_level["metrics/CoT"] = metrics_CoT

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return metrics_velrmsd


TASK_NAME = "Flat-Lus2"
DEVICE = "cuda:0"
if __name__ == "__main__":
    # 1)
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Run Optuna study with Neptune logging.")
    parser.add_argument("--task", type=str, default="Flat-Lus2", help="Task name for the study.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to be used for training.")
    args = parser.parse_args()

    TASK_NAME = args.task
    DEVICE = args.device

    print("The task name in this study is: ", TASK_NAME)
    print("The device being used is: ", DEVICE)

    # 2) create neptune study level run
    run_study_level = neptune.init_run(
        project="suntao/ambot-v2-rl",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        mode="offline",
    )
    # 3) create neptune_callback for optuna
    neptune_callback = optuna_utils.NeptuneCallback(run_study_level)
    study_name = TASK_NAME + "_rl_hyperparam"
    storage_name = "sqlite:///{}.db".format(study_name)

    # 4) creat optuna study
    # study = optuna.create_study(directions=["minimize","minimize"]) # multi object do not support multi objective
    if not os.path.exists("./../../{}.db".format(study_name)):  # creating study
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
        )
    else:  # loading study
        study = optuna.load_study(study_name=study_name, storage=storage_name)

    # optuna study and call neptune callback
    study.optimize(objective, n_trials=100, callbacks=[neptune_callback])
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # stop neptune
    run_study_level.stop()
