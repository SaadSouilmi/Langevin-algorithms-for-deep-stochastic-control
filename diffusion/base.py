from abc import ABC, abstractmethod


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

    @abstractmethod
    def set_control(self):
        """Abstract method that defines the control"""
        pass

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
