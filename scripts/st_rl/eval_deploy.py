import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
import re
from collections import defaultdict
from datetime import datetime


# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eval_play_and_deployment")

# -------------------- Argument Parsing --------------------
parser = argparse.ArgumentParser(description="Evaluate/Assess an RL agent with RSL-RL.")
parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment folder where logs are stored.")
parser.add_argument("--load_run", type=str, required=True, help="Name of the run folder to resume from.")
parser.add_argument("--nohup", type=str, default="nohup.out", help="Nohup output file to extract from.")
args_cli, _ = parser.parse_known_args()

# -------------------- Extract Deploy Log --------------------
def extract_deploy_log(log_file_name):
    """
    Parses a deployment log (nohup) file and extracts numeric values for various variables.
    """
    #log_path = f"/home/thomas/workspace/lumos_ws/lumos_controller/build/{log_file_name}"
    log_path = os.path.abspath(os.path.join("logs", "st_rl", args_cli.experiment_name, args_cli.load_run, f"nohup/{log_file_name}"))
    logger.info(f"Reading deploy log from: {log_path}")
    if not os.path.exists(log_path):
        logger.info(f"Not Exist: {log_path}")
        return None

    variables_to_extract = [
        "actions_", "test_action", "last action", "target_q_", "input_ob_", "test_obs", "tau_",
        "goal ref_motion", "rl_iter_", "cmd_", "q_", "qd_", "omega", "acc", "rpy", "projected_gravity", "frame_idx"
    ]

    patterns = {var: re.compile(rf"\s+{re.escape(var)}:\s+([-\d.eE+ ]+)") for var in variables_to_extract}
    extracted_data = defaultdict(list)

    started = False  # Flag to start parsing only after marker

    with open(log_path, "r") as file:
        for line in file:
            if not started and ("rl_iter" in line or "cur_iter_" in line):
                started = True
                logger.info("Found 'rl_iter' marker — starting data extraction.")

            if not started:
                continue  # Skip lines until the marker is found

            for var, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    try:
                        numbers = [float(x) for x in match.group(1).strip().split()]
                        extracted_data[var].append(numbers)
                    except ValueError:
                        logger.warning(f"Failed to parse values for {var}: {line.strip()}")

    return {k: np.array(v) for k, v in extracted_data.items()}

# -------------------- Load MuJoCo Evaluation Logs --------------------
def load_mujoco_log(args):
    """
    Loads MuJoCo exported evaluation results and joint mappings.
    """
    log_root = os.path.abspath(os.path.join("logs", "st_rl", args.experiment_name, args.load_run))
    eval_folder = os.path.join(log_root, "exported")
    logger.info(f"Loading MuJoCo evaluation data from: {eval_folder}")

    data = {}
    file_map = {
        "store_mj_onnx_obs.txt": "mj_onnx_obs",
        "store_mj_onnx_action.txt": "mj_onnx_action",
        "store_mj_rknn_action.txt": "mj_rknn_action",
        "store_mj_onnx_target_q.txt": "mj_onnx_target_q",
        "store_ref_motion.txt": "mj_ref_motion"
    }

    for filename, key in file_map.items():
        path = os.path.join(eval_folder, filename)
        if os.path.isfile(path):
            try:
                data[key] = np.loadtxt(path, delimiter=" ")
                logger.info(f"Loaded {key} from {filename}")
            except Exception as e:
                logger.warning(f"Could not load {key}: {e}")
        else:
            logger.warning(f"Missing file: {filename}")
            data[key] = None

    # Load joint name mappings
    joint_yaml_path = os.path.join(eval_folder, "joint_names.yaml")
    if os.path.isfile(joint_yaml_path):
        try:
            with open(joint_yaml_path, "r") as f:
                joint_data = yaml.safe_load(f)
                data["robot_joint_names"] = joint_data.get("robot_joint_names", [])
                data["policy_joint_names"] = joint_data.get("policy_joint_names", [])
        except Exception as e:
            logger.error(f"Failed to load joint_names.yaml: {e}")
            data["robot_joint_names"] = []
            data["policy_joint_names"] = []
    else:
        logger.warning(f"Missing joint_names.yaml in {eval_folder}")
        data["robot_joint_names"] = []
        data["policy_joint_names"] = []


    # Load ref motion fields
    ref_motion_field_yaml_path = os.path.join(eval_folder, "ref_motion_fields.yaml")
    if os.path.isfile(ref_motion_field_yaml_path):
        try:
            with open(ref_motion_field_yaml_path, "r") as f:
                fields = yaml.safe_load(f)
                data["ref_motion_fields"] = fields.get("ref_motion_fields", [])
        except Exception as e:
            logger.error(f"Failed to load ref_motion_fields.yaml: {e}")
            data["ref_motion_fields"] = []
    else:
        logger.warning(f"Missing ref_motion_fields.yaml in {eval_folder}")
        data["ref_motion_fields"] = []

    # Load kp kd
    kpkd_yaml_path = os.path.join(eval_folder, "kp_kd.yaml")
    if os.path.isfile(kpkd_yaml_path):
        try:
            with open(kpkd_yaml_path, "r") as f:
                kpkd_data = yaml.safe_load(f)
                data["kps"] = kpkd_data.get("kps", [])
                data["kds"] = kpkd_data.get("kds", [])
        except Exception as e:
            logger.error(f"Failed to load kp_kd.yaml: {e}")
            data["kps"] = []
            data["kds"] = []
    else:
        logger.warning(f"Missing kp_kd.yaml in {eval_folder}")
        data["kps"] = []
        data["kds"] = []

    # loading extras
    extras_data_path = os.path.join(eval_folder, "store_extras.pkl")
    import joblib
    extras = joblib.load(extras_data_path)  # Load
    # loading joint names from mujoco
    #policy_dof_index = [robot_joint_names.index(key) for key in policy_joint_names]
    robot_joint_index = [data["policy_joint_names"].index(key) for key in data["robot_joint_names"]]
    data["joint_pos"] = np.array([tmp["joint_pos"][robot_joint_index]  for tmp in extras])
    data["joint_vel"] = np.array([tmp["joint_vel"][robot_joint_index]  for tmp in extras])
    data["joint_tor"] = np.array([tmp["joint_tor"]  for tmp in extras])

    data["joint_power"] = data["joint_tor"] * data["joint_vel"]

    data["l_grf"] = np.array([tmp["grf"][0]  for tmp in extras])
    data["r_grf"] = np.array([tmp["grf"][1]  for tmp in extras])
    logger.info(f"mj data joint pos shape: {data['joint_pos'].shape}")

    return data


import matplotlib.pyplot as plt

def plot_joint_timeseries(
    joint_names,
    data_dicts,
    title,
    fields,  # e.g., ["q_", "target_q_"]
    frame_start=0,
    frame_end=1000,
    fps=50,
    ylabel="Value",
    indices=None,
    extra_lines_fn=None,
    fn_paras=None,
    args=None
):
    num_joints = len(indices) if indices is not None else len(joint_names)
    indices = indices if indices is not None else list(range(num_joints))
    cmap = plt.get_cmap("tab10")
    time = np.linspace(frame_start / fps, frame_end / fps, frame_end - frame_start)

    fig, axs = plt.subplots(num_joints, 1, figsize=(12, 2.2 * num_joints), sharex=True)
    fig.suptitle(title, fontsize=16)

    for plot_idx, joint_idx in enumerate(indices):
        joint_name = joint_names[joint_idx]
        ax = axs[plot_idx]

        if isinstance(fields, str):
            fields = [fields]

        for data, field_name in zip(data_dicts, fields):
            if field_name not in data:
                print(f"field name '{field_name}' does not exist in {list(data.keys())}")
                return

            if field_name in ["goal ref_motion", "mj_ref_motion"]:
                # Map to ref_motion_fields index
                if "ref_motion_fields" not in data:
                    raise ValueError(f"'ref_motion_fields' missing from data for field '{field_name}'")

                field_joint_key = joint_names[joint_idx] + "_dof_pos"
                if field_joint_key not in data["ref_motion_fields"]:
                    raise ValueError(f"'{field_joint_key}' not in ref_motion_fields for field '{field_name}'")

                field_joint_idx = data["ref_motion_fields"].index(field_joint_key)
            else:
                field_joint_idx = joint_idx

            if frame_end > data[field_name].shape[0]:
                print(f"frame_end: {frame_end} is larger than data[{field_name}] length: {data[field_name].shape[0]}, set it to be {data[field_name].shape[0]}")
                frame_end = data[field_name].shape[0]
                time = np.linspace(frame_start / fps, frame_end / fps, frame_end - frame_start)

            if field_name in ["l_grf", "r_grf"]:
                for ii in range(3):
                    grf = data[field_name][frame_start:frame_end, ii]
                    ax.plot(time, grf, label=f'mj {field_name} [{ii}]', marker=".", markersize=3)
            else:
                ax.plot(
                    time,
                    data[field_name][frame_start:frame_end, field_joint_idx],
                    label=f"{field_name}[{joint_name}] {field_joint_idx}",
                    linestyle="-",
                    markersize=3
                )


        if extra_lines_fn is not None:
            if fn_paras is None:
                extra_lines_fn(ax, time, joint_idx, joint_name, data)
            else:
                extra_lines_fn(ax, time, joint_idx, joint_name, data, fds=fn_paras)

        ax.set_ylabel(f"{joint_name}\n{ylabel}")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.minorticks_on()
        ax.legend(loc="upper right")

    axs[-1].set_xlabel("Time [s]")
    plt.tight_layout(rect=[0, 0, 1, 0.96])


    filename = f"{title.replace(' ', '_')}.png"
    if args is not None:
        log_root = os.path.abspath(os.path.join("logs", "st_rl", args.experiment_name, args.load_run))
    else:
        log_root = os.path.abspath(os.path.join("logs", "st_rl", args_cli.experiment_name, args_cli.load_run))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_dir = os.path.join(log_root, "figures", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, title)
    fig.savefig(fig_path)




# Plot real robot tau using subplots
def extra_tau_lines(ax, time, joint_idx, joint_name, data, fds=["target_q_","q_","qd_"]):

    kps = mj_eval_data["kps"]
    kds = mj_eval_data["kds"]
    
    tau_calc = (
        kps[joint_idx] * (
            data[fds[0]][frame_start:frame_end, joint_idx]
            - data[fds[1]][frame_start:frame_end, joint_idx]
        ) - kds[joint_idx] * data[fds[2]][frame_start:frame_end, joint_idx]
    )
    ax.plot(
        time, tau_calc,
        label=f"Calculated tau[{joint_name}]",
        linestyle="--",
        marker=".",
        markersize=4
    )


# -------------------- Main Execution --------------------
if __name__ == "__main__":
    # Load logs
    realrobot_nohup_data = extract_deploy_log(args_cli.nohup)
    mj_eval_data = load_mujoco_log(args_cli)

    robot_joint_names = mj_eval_data["robot_joint_names"]
    policy_joint_names = mj_eval_data["policy_joint_names"]
    frame_start = 0
    frame_end = 1000
    fps=50
    start_joint_idx = 0
    num_joints = 6

    # -------------------- Plotting Real Robot --------------------
    if realrobot_nohup_data is not None:
        # Plot real robot actions with subplots
        plot_joint_timeseries(
        joint_names=robot_joint_names,
        data_dicts=[realrobot_nohup_data],
        title="Real Robot actions Over Time",
        fields="actions_",
        frame_start = frame_start,
        frame_end = frame_end,
        fps=fps,
        ylabel="Action",
        indices=range(start_joint_idx, start_joint_idx + num_joints)
        )

        # Plot real robot q with subplots
        plot_joint_timeseries(
        joint_names=robot_joint_names,
        data_dicts=[realrobot_nohup_data],
        title="Real Robot q Over Time",
        fields="q_",
        frame_start = frame_start,
        frame_end = frame_end,
        fps=fps,
        ylabel="Position [rad]",
        indices=range(start_joint_idx, start_joint_idx + num_joints)
        )


        # Plot real robot dq with subplots
        plot_joint_timeseries(
        joint_names=robot_joint_names,
        data_dicts=[realrobot_nohup_data],
        title="Real Robot qd Over Time",
        fields="qd_",
        frame_start = frame_start,
        frame_end = frame_end,
        fps=fps,
        ylabel="Velocity [rad/s]",
        indices=range(start_joint_idx, start_joint_idx + num_joints)
        )


        # Plot real robot target q with subplots
        plot_joint_timeseries(
        joint_names=robot_joint_names,
        data_dicts=[realrobot_nohup_data],
        title="Real Robot Target q Over Time",
        fields="target_q_",
        frame_start = frame_start,
        frame_end = frame_end,
        fps=fps,
        ylabel="Target Position [rad]",
        indices=range(start_joint_idx, start_joint_idx + num_joints)
        )


        # plot real robot tau 
        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[realrobot_nohup_data],
            title="Real Robot Tau Over Time",
            fields="tau_",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="Torque [Nm]",
            indices=range(start_joint_idx, start_joint_idx + num_joints),
            extra_lines_fn=extra_tau_lines
        )


        # plot real robot projected gravity
        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[realrobot_nohup_data],
            title="Real robot grav Over Time",
            fields="projected_gravity",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="Pro gravity",
            indices=range(0, 3),
        )


        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[realrobot_nohup_data],
            title="Real robot Omega Over Time",
            fields="omega",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="Omega",
            indices=range(0, 3),
        )

        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[realrobot_nohup_data],
            title="Real robot Acc Over Time",
            fields="acc",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="Acceleration",
            indices=range(0, 3),
        )

        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[realrobot_nohup_data],
            title="Real robot RPY Over Time",
            fields="rpy",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="Euler",
            indices=range(0, 3),
        )


        # Plot real robot q and ref_motion
        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[realrobot_nohup_data, mj_eval_data],
            fields=["q_","mj_ref_motion"],
            title="Real q and ref_q Over Time",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="q and ref_q [rad]",
            indices=range(start_joint_idx, start_joint_idx + num_joints),
        )


    # -------------------- Plotting Mujoco  --------------------
    if mj_eval_data is not None:
        #frame_start=1
        #frame_end = 2000
        # Plot mj robot with subplots
        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[mj_eval_data, mj_eval_data],
            fields=["joint_pos","mj_ref_motion"],
            title="MJ  q and ref_q Over Time",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="MJ q and ref_q [rad]",
            indices=range(start_joint_idx, start_joint_idx + num_joints),
        )


        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[mj_eval_data, mj_eval_data],
            fields=["joint_pos","mj_ref_motion"],
            title="MJ  q and ref_q Over Time",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="MJ q and ref_q [rad]",
            indices=range(start_joint_idx+12, start_joint_idx + num_joints+12),
        )

        # Plot mj actions
        plot_joint_timeseries(
        joint_names=robot_joint_names,
        data_dicts=[mj_eval_data, mj_eval_data],
        fields="mj_onnx_action",
        title="MJ Actions Over Time",
        frame_start = frame_start,
        frame_end = frame_end,
        fps=fps,
        ylabel="Action",
        indices=range(start_joint_idx, start_joint_idx + num_joints)
        )

        # Plot mj dof pos
        plot_joint_timeseries(
        joint_names=robot_joint_names,
        data_dicts=[mj_eval_data, mj_eval_data],
        title="MJ Dof Pos Over Time",
        fields="joint_pos",
        frame_start = frame_start,
        frame_end = frame_end,
        fps=fps,
        ylabel="joint pos [rad]",
        indices=range(start_joint_idx, start_joint_idx + num_joints)
        )

        # Plot mj dof vel
        plot_joint_timeseries(
        joint_names=robot_joint_names,
        data_dicts=[mj_eval_data, mj_eval_data],
        title="MJ Dof Vel Over Time",
        fields="joint_vel",
        frame_start = frame_start,
        frame_end = frame_end,
        fps=fps,
        ylabel="joint vel",
        indices=range(start_joint_idx, start_joint_idx + num_joints)
        )

        # Plot mj dof tor
        plot_joint_timeseries(
        joint_names=robot_joint_names,
        data_dicts=[mj_eval_data],
        title="Mj Dof torque Over Time",
        fields="joint_tor",
        frame_start = frame_start,
        frame_end = frame_end,
        fps=fps,
        ylabel="joint torque",
        indices=range(start_joint_idx, start_joint_idx + num_joints),
        extra_lines_fn=extra_tau_lines,
        fn_paras=["mj_onnx_target_q","joint_pos","joint_vel"]
        )


        # Plot mj power
        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[mj_eval_data],
            title="MJ Power Over Time",
            fields="joint_power",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="Power [W]",
            indices=range(start_joint_idx, start_joint_idx + num_joints),
        )


        # Plot mj GRF
        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[mj_eval_data],
            title="MJ left GRF Over Time",
            fields="l_grf",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="GRF",
            indices=range(0,2),
        )
        # Plot mj GRF
        plot_joint_timeseries(
            joint_names=robot_joint_names,
            data_dicts=[mj_eval_data],
            title="MJ left GRF Over Time",
            fields="r_grf",
            frame_start = frame_start,
            frame_end = frame_end,
            fps=fps,
            ylabel="GRF",
            indices=range(0,2),
        )

        plt.show()

