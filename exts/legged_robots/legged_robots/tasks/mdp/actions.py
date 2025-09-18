from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions import JointAction, JointPositionAction, JointActionCfg, JointPositionActionCfg

from isaaclab.envs import ManagerBasedEnv
import torch




class StJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self.position_noise_std = cfg.position_noise_std


    def apply_actions(self):

        # Add effort noise before applying
        if self.position_noise_std > 0.0:
            noise = torch.randn_like(self.processed_actions) * self.position_noise_std
        else:
            noise = 0.0

        # set position targets
        self._asset.set_joint_position_target(self.processed_actions+noise, joint_ids=self._joint_ids)



@configclass
class StJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = StJointPositionAction

    use_default_offset: bool = True
    position_noise_std: float = 0.0
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
