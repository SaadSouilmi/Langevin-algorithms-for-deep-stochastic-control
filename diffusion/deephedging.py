from .base import DeepControledDiffusion
import torch
import math
import tqdm
from typing import Callable


class DeepHedging(DeepControledDiffusion):
    """This class implements the deep hedging problem"""

    def __init__(
        self,
        T: float = 50.0 / 365.0,
        N_euler: int = 50,
        dim: int = 1,
        ell: Callable = lambda x: x,
        a: torch.Tensor = 0.3 * torch.ones(1),
        b: torch.Tensor = 0.04 * torch.ones(1),
        sigma: torch.Tensor = 2.0 * torch.ones(1),
        rho: torch.Tensor = 1.0 * torch.ones(1),
        K: torch.Tensor = 1.0 * torch.ones(1),
        T_COST: float = 0.01,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__(T, N_euler, dim)
        self.ell = ell
        self.a = a
        self.b = b
        self.sigma = sigma
        self.rho = rho
        self.K = K
        self.T_COST = T_COST
        self.epsilon = epsilon

    def L(self, t: float, v: torch.Tensor) -> torch.Tensor:
        """Helper function that computes L
        Args:
            - t: time
            - v: variance
        Returns:
            - torch.Tensor: L"""
        return (
            v.add(self.b, alpha=-1)
            .div(self.a)
            .mul(1 - torch.exp(-(self.T - t) * self.a))
            + (self.T - t) * self.b
        )

    def sample_traj(
        self,
        batch_size: int,
        S_0: torch.Tensor,
        V_0: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Samples X_0 from a gaussian clipped to [lbound, ubound]
        Args:
            - batch_size: number of initial conditions to sample
            - S_0: initial value of the spot
            - V_0: initial value of the variance
            - device: torch device"""
        pass
