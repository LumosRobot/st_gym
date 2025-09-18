import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process
import pandas as pd
import collections
from termcolor import colored
import os
import logging
from typing import Dict, List, Union, Optional
from contextlib import contextmanager
import threading
from dataclasses import dataclass
import warnings
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class PlotConfig:
    """Configuration for plotting parameters"""
    figsize: tuple = (26, 18)
    font_size: int = 18
    dpi: int = 150
    grid: bool = True

class DataBuffer:
    """Buffer for managing large datasets efficiently"""
    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size
        self.buffer = defaultdict(lambda: np.zeros(max_size))
        self.current_idx = 0
        self._lock = threading.Lock()

    def add(self, key: str, value: np.ndarray) -> None:
        """Add data to the buffer"""
        with self._lock:
            if self.current_idx >= self.max_size:
                self._resize_buffer()
            self.buffer[key][self.current_idx] = value

    def _resize_buffer(self) -> None:
        """Resize the buffer when it exceeds the maximum size"""
        new_size = self.max_size * 2
        for key in self.buffer:
            self.buffer[key] = np.resize(self.buffer[key], new_size)
        self.max_size = new_size

    def get_data(self, key: str) -> np.ndarray:
        """Get data from the buffer"""
        return self.buffer[key][:self.current_idx]

class Logger:
    """Enhanced logger with efficient data handling and visualization capabilities"""
    
    def __init__(self, 
                 dt: float, 
                 log_dir: Optional[str] = None, 
                 show_figure: bool = False,
                 idx = None, 
                 buffer_size: int = 1000000):
        """
        Initialize the logger with given parameters.
        
        Args:
            dt: Time step between logging events
            log_dir: Directory for saving logs
            show_figure: Whether to display figures
            idx: Logger identifier
            buffer_size: Maximum size of data buffer
        """
        self.log_dir = self._validate_log_dir(log_dir)
        # Auto-determine idx if not provided

        self.logger = logging.getLogger(f"States_{idx}")

        self.log_id = self._get_next_idx() if idx is None else idx

        self.data_buffer = DataBuffer(buffer_size)
        self.reward_buffer = DataBuffer(buffer_size)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.show_figure = show_figure

        self.plot_config = PlotConfig()
        
        # Define constants
        self.COLORS_MAP = collections.OrderedDict({
            "measured_x": "r-", "command_x": "r-.",
            "measured_y": "g-", "command_y": "g-.",
            "measured_z": "b-", "command_z": "b-.",
        })

    def _get_next_idx(self) -> int:
        """
        Determine the next available index based on existing state_*.csv files.
        Returns the highest existing index + 1, or 0 if no files exist.
        """
        try:
            # List all files in log directory
            files = os.listdir(self.log_dir)
            
            # Find all state_*.csv files
            state_files = [f for f in files if f.startswith('states_') and f.endswith('.csv')]
            
            if not state_files:
                return 0
                
            # Extract indices using regex
            indices = []
            pattern = re.compile(r'states_(\d+)\.csv')
            for file in state_files:
                match = pattern.match(file)
                if match:
                    indices.append(int(match.group(1)))
            
            # Return next available index
            return max(indices) + 1 if indices else 0
            
        except Exception as e:
            import pdb;pdb.set_trace()
            self.logger.error(f"Error determining next index: {str(e)}")
            return 0

    @staticmethod
    def _validate_log_dir(log_dir: Optional[str]) -> str:
        """Validate and create log directory if needed"""
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @contextmanager
    def error_handling(self, operation: str):
        """Context manager for error handling"""
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error during {operation}: {str(e)}")
            raise

    def update_state(self, key: str, value: Union[float, np.ndarray]) -> None:
        """Log a single state value with validation"""
        try:
            value_np = np.array(value)
            self.data_buffer.add(key, value_np)
        except ValueError as e:
            self.logger.error(f"Invalid value for key {key}: {str(e)}")

    def update_states(self, state_dict: Dict[str, Union[float, np.ndarray]]) -> None:
        """Log multiple states with validation"""
        with self.error_handling("logging states"):
            for key, value in state_dict.items():
                self.update_state(key, value)
            self.data_buffer.current_idx += 1

    def log_rewards(self, reward_dict: Dict[str, float], num_episodes: int) -> None:
        """Log rewards with validation"""
        with self.error_handling("logging rewards"):
            for key, value in reward_dict.items():
                if 'rew' in key:
                    try:
                        value_np = np.array(value.item() * num_episodes)
                        self.reward_buffer.add(key, value_np)
                    except (ValueError, AttributeError) as e:
                        self.logger.error(f"Invalid reward value for key {key}: {str(e)}")
            self.num_episodes += num_episodes

    def reset(self) -> None:
        """Reset all buffers"""
        self.data_buffer = DataBuffer()
        self.reward_buffer = DataBuffer()
        self.num_episodes = 0

    def save(self, file_path: Optional[str] = None) -> None:
        """Store states to CSV file asynchronously"""
        if file_path is None:
            file_path = self.log_dir
        self.store_process = Process(target=self._store, args=(file_path,))
        self.store_process.start()

    def _store(self, file_path: str) -> None:
        """Helper method to store states to CSV with error handling"""
        with self.error_handling("storing states"):
            data_dict = {key: self.data_buffer.get_data(key) 
                        for key in self.data_buffer.buffer.keys()}
            pd_data = pd.DataFrame(data_dict)
            
            output_path = os.path.join(file_path, f"states_{self.log_id}.csv")
            with open(output_path, 'w') as f:
                pd_data.to_csv(f)
            print("[Info]: Data saved to ", output_path)

    def _setup_plot(self) -> tuple:
        """Setup plot configuration with error handling"""
        with self.error_handling("setting up plot"):
            plt.rcParams.update({'font.size': self.plot_config.font_size})
            return plt.subplots(5, 4, figsize=self.plot_config.figsize)

    def _get_time_array(self) -> np.ndarray:
        """Generate time array for plotting"""
        for key in self.data_buffer.buffer:
            data = self.data_buffer.get_data(key)
            return np.linspace(0, len(data) * self.dt, len(data))
        return np.array([])

    def plot_states(self) -> None:
        """Plot states asynchronously"""
        self.plot_process = Process(target=self._plot1)
        self.plot_process.start()

    # ... rest of plotting methods would be similarly enhanced ...

    def _plot1(self):
        """Plotting method for states"""
        nb_rows = 5
        nb_cols = 4
        import matplotlib
        plt.rcParams.update({'font.size': 18})
        fig, axs = plt.subplots(nb_rows, nb_cols)
        fig.set_figheight(18)
        fig.set_figwidth(26)
        plt.subplots_adjust(top=0.94, bottom=0.05, left=0.06, right=0.97, hspace=0.38,
                    wspace=0.18)
        # ...existing code...

        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        colors_map=collections.OrderedDict({
                "measured_x": "r-", "command_x": "r-.",
                "measured_y": "g-", "command_y": "g-.",
                "measured_z": "b-", "command_z": "b-.",
                })
        # plot base vel x
        a = axs[0, 0]
        if log["base_lin_vel_x"]: a.plot(time, log["base_lin_vel_x"], colors_map["measured_x"], label=r'$v_x$')
        if log["command_x"]: a.plot(time, log["command_x"], colors_map["command_x"], label=r'$cmd_x$')
        if log["base_lin_vel_y"]: a.plot(time, log["base_lin_vel_y"], colors_map["measured_y"], label=r'$v_y$')
        if log["command_y"]: a.plot(time, log["command_y"], colors_map["command_y"],label=r'$cmd_y$')
        if log["base_lin_vel_z"]: a.plot(time, log["base_lin_vel_z"], colors_map["measured_z"], label=r'$v_z$')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], colors_map["command_z"], label=r'$cmd_z$')
        a.set(xlabel='Time [s]', ylabel=r'$v_b$ [m/s]', title='Base linear velocity')
        a.legend(loc="upper right")
        a.grid()

        # plot base ang vel x, y, and z
        a = axs[0, 1]
        if log["base_ang_vel_x"]: a.plot(time, log["base_ang_vel_x"], colors_map["measured_x"], label=r'$w_x$')
        if log["base_ang_vel_y"]: a.plot(time, log["base_ang_vel_y"], colors_map["measured_y"], label=r'$w_y$')
        if log["base_ang_vel_z"]: a.plot(time, log["base_ang_vel_z"], colors_map["measured_z"], label=r'$w_z$')
        a.set(xlabel='Time [s]', ylabel=r'$w_b$ [rad/s]', title='Base angular velocity')
        a.legend(loc="upper right")
        a.grid()

        # plot base orientation
        a = axs[0, 2]
        if log["base_pitch"]:
            a.plot(time, log["base_pitch"], colors_map["measured_x"],label='Pitch')
            a.plot(time, [-0.75] * len(time), label= 'Threshold')
        if log["base_roll"]: a.plot(time, log["base_roll"], colors_map["measured_y"], label='Roll')
        if log["base_yaw"]:  a.plot(time, log["base_yaw"], colors_map["measured_z"], label='Yaw')
        # if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='Time [s]', ylabel='$Orientation$ [rad]', title='Base orientation')
        a.legend(loc="upper right")
        a.grid()

        # plot base projected gravity x, y, and z
        a = axs[0, 3]
        if log["base_pro_gravity_x"]: a.plot(time, log["base_pro_gravity_x"], colors_map["measured_x"], label=r'$g_x$')
        if log["base_pro_gravity_y"]: a.plot(time, log["base_pro_gravity_y"], colors_map["measured_y"], label=r'$g_y$')
        if log["base_pro_gravity_z"]: a.plot(time, log["base_pro_gravity_z"], colors_map["measured_z"], label=r'$g_z$')
        a.set(xlabel='Time [s]', ylabel=r'$g_{pro}$', title='Base projected gravity vector')
        a.legend(loc="upper right")
        a.grid()

        # plot actions
        for ax_col in range(4): # axs col
            a = axs[1, ax_col]
            idx = 3*ax_col
            if log["action_"+str(idx)]: a.plot(time, log["action_"+str(idx)], label=r'$a_'+str(idx)+'$')
            if log["action_"+str(idx+1)]: a.plot(time, log["action_"+str(idx+1)], label=r'$a_'+str(idx+1)+'$')
            if log["action_"+str(idx+2)]: a.plot(time, log["action_"+str(idx+2)], label=r'$a_'+str(idx+2)+'$')
            a.set(xlabel='Time [s]', ylabel=r'$Actions$', title='Actions')
            a.legend(loc="upper right")
            a.grid()

        # plot joint dof desired pos and actual pos
        for ax_col in range(4): # axs col
            a = axs[2, ax_col]
            idx= 3*ax_col
            if log["dof_pos_"+str(idx)]: a.plot(time, log["dof_pos_"+str(idx)], 
                    #colors_map[list(colors_map.keys())[2*(idx-9)]], 
                    label=r'$q_'+str(idx)+"$")
            if log["dof_pos_target_"+str(idx)]: a.plot(time, log["dof_pos_target_"+str(idx)], 
                    #colors_map[list(colors_map.keys())[2*(idx-9)+1]], 
                    label=r'$q^{des}_{'+str(idx)+'}$')
            if log["dof_pos_"+str(idx+1)]: a.plot(time, log["dof_pos_"+str(idx+1)], label=r'$q_'+str(idx+1)+"$")
            if log["dof_pos_target_"+str(idx+1)]: a.plot(time, log["dof_pos_target_"+str(idx+1)], label=r'$q^{des}_{'+str(idx+1)+'}$')
            if log["dof_pos_"+str(idx+2)]: a.plot(time, log["dof_pos_"+str(idx+2)], label=r'$q_'+str(idx+2)+"$")
            if log["dof_pos_target_"+str(idx+2)]: a.plot(time, log["dof_pos_target_"+str(idx+2)], label=r'$q^{des}_{'+str(idx+2)+'}$')
            a.set(xlabel='Time [s]', ylabel=r'$\mathbf{q}$ [rad]', title='DOF Position')
            a.legend(loc="upper right")
            a.grid()

        # plot joint velocity
        for ax_col in range(4): # axs col
            a = axs[3, ax_col]
            idx = 3*ax_col
            if log["dof_vel_"+str(idx)]: a.plot(time, log["dof_vel_"+str(idx)], label=r'$\dot{q}_'+str(idx)+'$')
            if log["dof_vel_"+str(idx+1)]: a.plot(time, log["dof_vel_"+str(idx+1)], label=r'$\dot{q}_'+str(idx+1)+'$')
            if log["dof_vel_"+str(idx+2)]: a.plot(time, log["dof_vel_"+str(idx+2)], label=r'$\dot{q}_'+str(idx+2)+'$')
            a.set(xlabel='Time [s]', ylabel=r'$\mathbf{\dot{q}} [rad/s]$', title='Joint Velocity')
            a.legend(loc="upper right")
            a.grid()

        # plot dof torques
        a = axs[4, 0]
        for idx in range(3):
            if log["dof_torques_"+str(idx)]: a.plot(time, log["dof_torques_"+str(idx)], 
                    label=r'$\tau_'+str(idx)+'$')
        a.set(xlabel='Time [s]', ylabel=r'$\tau$ [Nm]', title='Joint Torque')
        a.legend(loc="upper right")
        a.grid()

        a = axs[4, 1]
        for idx in range(6,9):
            if log["torques_"+str(idx)]: a.plot(time, log["torques_"+str(idx)], 
                    label=r'$\tau_'+str(idx)+'$')
        a.set(xlabel='Time [s]', ylabel=r'$\tau$ [Nm]', title='Joint Torque')
        a.legend(loc="upper right")
        a.grid()

        a = axs[4, 2]
        for idx in range(0,3):
            if log["joint_pos_err_"+str(idx)]: a.plot(time, log["joint_pos_err_"+str(idx)], 
                    label='measured_'+str(idx))
        a.set(xlabel='Time [s]', ylabel='Joint position error [rad]', title='Joint pos err')
        a.legend(loc="upper right")
        a.grid()

        # plot contact forces
        a = axs[4, 3]
        for idx in range(4):
            if log["grf_z_"+str(idx)]:
                a.plot(time, log["grf_z_"+str(idx)], label=f'GRF {idx}')
        a.set(xlabel='Time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend(loc="upper right")
        a.grid()
        # # plot torque/vel curves
        # a = axs[2, 1]
        # if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        # a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        # a.legend()
        # plot power curves
        """
        a = axs[3, 3]
        if log["power"]!=[]: a.plot(time, log["power"], label='power [W]')
        a.set(xlabel='time [s]', ylabel='Power [W]', title='Power')
        
        # plot rewards
        a = axs[3, 0]
        if log["max_torques"]: a.plot(time, log["max_torques"], label='max_torques')
        if log["max_torque_motor"]: a.plot(time, log["max_torque_motor"], label='max_torque_motor')
        if log["max_torque_leg"]: a.plot(time, log["max_torque_leg"], label='max_torque_leg')
        a.set(xlabel='time [s]', ylabel='max_torques [Nm]', title='max_torques')
        a.legend(fontsize= 5)
        # plot customed data
        a = axs[3, 1]
        if log["student_action"]:
            a.plot(time, log["student_action"], label='s')
            a.plot(time, log["teacher_action"], label='t')
        a.legend()
        a.set(xlabel='time [s]', ylabel='value before step()', title='student/teacher action')
        a = axs[3, 2]
        a.plot(time, log["reward"], label='rewards')
        for i in log["mark"]:
            if i > 0:
                a.plot(time, log["mark"], label='user mark')
                break
        for key in log.keys():
            if "reward_removed_" in key:
                a.plot(time, log[key], label= key)
        a.set(xlabel='time [s]', ylabel='', title='rewards')
        # a.set_ylim([-0.12, 0.1])
        a.legend(fontsize = 3)
        """
        if self.show_figure:
            plt.show()
        plt.savefig(os.path.join(self.log_dir,"state_{}.svg".format(self.log_id)),dpi=150)
        print(colored(f"save fig at {self.log_dir}","green"))

    def _plot(self):
        nb_rows = 4
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base pitch
        a = axs[0, 2]
        if log["base_pitch"]:
            a.plot(time, log["base_pitch"], label='measured')
        # if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang [rad]', title='Base pitch')
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # # plot torque/vel curves
        # a = axs[2, 1]
        # if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        # a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        # a.legend()
        # plot power curves
        a = axs[2, 1]
        if log["power"]!=[]: a.plot(time, log["power"], label='power [W]')
        a.set(xlabel='time [s]', ylabel='Power [W]', title='Power')
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        # plot rewards
        a = axs[3, 0]
        if log["max_torques"]: a.plot(time, log["max_torques"], label='max_torques')
        if log["max_torque_motor"]: a.plot(time, log["max_torque_motor"], label='max_torque_motor')
        if log["max_torque_leg"]: a.plot(time, log["max_torque_leg"], label='max_torque_leg')
        a.set(xlabel='time [s]', ylabel='max_torques [Nm]', title='max_torques')
        a.legend(fontsize= 5)
        # plot customed data
        a = axs[3, 1]
        if log["student_action"]:
            a.plot(time, log["student_action"], label='s')
            a.plot(time, log["teacher_action"], label='t')
        a.legend()
        a.set(xlabel='time [s]', ylabel='value before step()', title='student/teacher action')
        a = axs[3, 2]
        a.plot(time, log["reward"], label='rewards')
        for i in log["mark"]:
            if i > 0:
                a.plot(time, log["mark"], label='user mark')
                break
        for key in log.keys():
            if "reward_removed_" in key:
                a.plot(time, log[key], label= key)
        a.set(xlabel='time [s]', ylabel='', title='rewards')
        # a.set_ylim([-0.12, 0.1])
        a.legend(fontsize = 5)
        plt.show()

    def _plot_vel(self):
        log= self.state_log
        nb_rows = int(np.sqrt(log['all_dof_vel'][0].shape[0]))
        nb_cols = int(np.ceil(log['all_dof_vel'][0].shape[0] / nb_rows))
        nb_rows, nb_cols = nb_cols, nb_rows
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        
        # plot joint velocities
        for i in range(nb_rows):
            for j in range(nb_cols):
                if i*nb_cols+j < log['all_dof_vel'][0].shape[0]:
                    a = axs[i][j]
                    a.plot(
                        time,
                        [all_dof_vel[i*nb_cols+j] for all_dof_vel in log['all_dof_vel']],
                        label='measured',
                    )
                    a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title=f'Joint Velocity {i*nb_cols+j}')
                    a.legend()
                else:
                    break
        plt.show()

    def _plot_torque(self):
        log= self.state_log
        nb_rows = int(np.sqrt(log['all_dof_torque'][0].shape[0]))
        nb_cols = int(np.ceil(log['all_dof_torque'][0].shape[0] / nb_rows))
        nb_rows, nb_cols = nb_cols, nb_rows
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        
        # plot joint torques
        for i in range(nb_rows):
            for j in range(nb_cols):
                if i*nb_cols+j < log['all_dof_torque'][0].shape[0]:
                    a = axs[i][j]
                    a.plot(
                        time,
                        [all_dof_torque[i*nb_cols+j] for all_dof_torque in log['all_dof_torque']],
                        label='measured',
                    )
                    a.set(xlabel='time [s]', ylabel='Torque [Nm]', title=f'Joint Torque {i*nb_cols+j}')
                    a.legend()
                else:
                    break
        plt.show()

    def print_rewards(self) -> None:
        """Print average rewards with error handling"""
        with self.error_handling("printing rewards"):
            self.logger.info("Average rewards per second:")
            for key in self.reward_buffer.buffer:
                values = self.reward_buffer.get_data(key)
                mean = np.mean(values)
                self.logger.info(f" - {key}: {mean:.4f}")
            self.logger.info(f"Total number of episodes: {self.num_episodes}")

    def __del__(self) -> None:
        """Cleanup resources"""
        try:
            if self.plot_process is not None:
                self.plot_process.kill()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

        try:
            if self.store_process is not None:
                self.store_process.kill()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


