from abc import ABC, abstractmethod
from typing import Union
import torch
from training.network import MLP


class DeepControledDiffusion(ABC):
    """This class serves as a template for the deep stochastic control problems"""

    def __init__(
        self,
        T: float = 1.0,
        N_euler: int = 50,
        dim: int = 1,
    ) -> None:
        """Args:
        - T: diffusions time horizon
        - N_euler: number of discretized time steps
        - dim: output dimension of the control"""
        self.T = T
        self.N_euler = N_euler
        self.h = T / N_euler
        self.dim = dim

    def set_control(self, control: dict, multiple_controls: bool = False) -> None:
        """Function that sets the control
        Args:
            - control: dictionnary containing the config of the control
            - mutiple_controls: boolean determining whether we use mutiple controls or not
        """
        self.multiple_controls = multiple_controls
        if multiple_controls:
            self.control = [MLP(**control) for k in range(self.N_euler + 1)]
        else:
            self.control = MLP(**control)

    def train_mode(self) -> None:
        """Sets the control to train mode"""
        if not self.multiple_controls:
            self.control.train()
        else:
            for control in self.control:
                control.train

    def eval_mode(self) -> None:
        """Sets the control to eval mode"""
        if not self.multiple_controls:
            self.control.eval()
        else:
            for control in self.control:
                control.eval()

    @abstractmethod
    def sample_start(self):
        """Abstract method that samples the starting condition of the diffusion"""
        pass

    @abstractmethod
    def sample_traj(self):
        """Abstract method that samples trajectories of the diffusion"""
        pass

    @abstractmethod
    def objective(self):
        """Abstract method that computes the control objective J"""
        pass
