from base import DeepControledDiffusion
import torch
import math


class Fishing(DeepControledDiffusion):
    """This class implements the fishing quota problem"""

    def __init__(
        self,
        T: float = 1.0,
        N_euler: int = 50,
        dim: int = 5,
        multiple_controls: bool = False,
        r: torch.Tensor = 2 * torch.ones(5),
        kappa: torch.Tensor = torch.ones((5, 5)),
        X_d: torch.Tensor = torch.ones(5),
        u_m: float = 0.1,
        u_M: float = 1.0,
        alpha: torch.Tensor = 0.01 * torch.ones(5),
        beta: float = 0.1,
        sigma: float = 0.1 * torch.eye(5),
        init_mean: torch.Tensor = torch.ones(5),
        init_std: float = 0.5,
        init_lbound: float = 0.2,
        init_ubound: float = 2,
    ) -> None:

        super().__init__(T, N_euler, dim)
        self.r = r
        self.kappa_t = kappa.T
        self.X_d = X_d
        self.sigma_t = sigma.T
        self.u_m = u_m
        self.u_M = u_M
        self.alpha = alpha
        self.beta = beta
        self.multiple_controls = multiple_controls
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_lbound = init_lbound
        self.init_ubound = init_ubound

    def set_control(
        self, control: list[torch.nn.Module], device: torch.device = torch.device("cpu")
    ) -> None:
        """Function that sets the control
        Args:
            - control: list of all controls, contains one element in case of multiple controls
            - device: torch device"""
        if not self.multiple_controls:
            self.control = control[0]
        else:
            self.control = control

    def sample_start(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Samples X_0 from a gaussian clipped to [lbound, ubound]
        Args:
            - batch_size: number of initial conditions to sample
            - device: torch device"""
        # broadcast the mean tensor and cast to device
        batched_mean = (
            self.init_mean * torch.ones((batch_size, len(self.init_mean)))
        ).to(device)
        return torch.clamp(
            torch.normal(batched_mean, std=self.init_std),
            min=self.init_lbound,
            max=self.init_ubound,
        )

    def sample_traj(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples X_0 from a gaussian clipped to [lbound, ubound]
        Args:
            - batch_size: number of trajectories to sample
            - device: torch device
        Returns:
            - tuple[torch.Tensor, torch.Tensor]: tuple containing trajectory of X and corresponding control u
        """
        X_0 = self.sample_start(batch_size, device)
        pass

    def objective(self, batch_size: int, device) -> torch.Tensor:
        """Computes the control objective J
        Args:
            - batch_size: number of trajectories to sample
            - device: torch device
        Returns:
            - torch.Tensor: tensor J"""
        pass
