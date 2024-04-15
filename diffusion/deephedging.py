from .base import DeepControledDiffusion
import torch
import math
import tqdm
from typing import Callable
from training.network import MLP


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
        S_0: torch.Tensor = torch.ones(1),
        V_0: torch.Tensor = 0.1 * torch.ones(1),
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
        self.S_0 = S_0
        self.V_0 = V_0
        self.epsilon = epsilon
        self.name = "Deep hedging"

    def set_control(
        self, control_config: dict, multiple_controls: bool = False
    ) -> None:
        """Function that sets the control
        Args:
            - control_config: dictionnary containing the config of the control
            - mutiple_controls: boolean determining whether we use mutiple controls or not
        """
        self.multiple_controls = multiple_controls
        if multiple_controls:
            self.control = [MLP(**control_config) for k in range(self.N_euler)]
        else:
            self.control = MLP(**control_config)
        self.w = torch.nn.Linear(1, 1, bias=False)

    def train_mode(self) -> None:
        """Sets the control to train mode"""
        if not self.multiple_controls:
            self.control.train()
        else:
            for control in self.control:
                control.train()
        self.w.train()

    def eval_mode(self) -> None:
        """Sets the control to eval mode"""
        if not self.multiple_controls:
            self.control.eval()
        else:
            for control in self.control:
                control.eval()
        self.w.eval()

    def L(
        self, t: float, v: torch.Tensor, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Helper function that computes L
        Args:
            - t: time
            - v: variance
            - device: torch device
        Returns:
            - torch.Tensor: L"""
        return v.add(self.b.to(device), alpha=-1).div(self.a.to(device)).mul(
            1 - torch.exp(-(self.T - t) * self.a.to(device))
        ) + (self.T - t) * self.b.to(device)

    def sample_traj(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        disable_tqdm: bool = False,
    ) -> torch.Tensor:
        """Samples trajectory of S, V and u
        Args:
            - batch_size: number of initial conditions to sample
            - S_0: initial value of the spot
            - V_0: initial value of the variance
            - device: torch device
            - disable_tqdm: whether or not to disable progress bar"""
        # Batchify initial conditions
        S_0 = self.S_0.to(device) * torch.ones((batch_size, self.dim)).to(device)
        V_0 = self.V_0.to(device) * torch.ones((batch_size, self.dim)).to(device)
        # Initialize buffers
        V_buf = torch.zeros((batch_size, self.N_euler + 1, self.dim)).to(device)
        V_buf[:, 0, :] = V_0
        S1_buf = torch.zeros((batch_size, self.N_euler + 1, self.dim)).to(device)
        S1_buf[:, 0, :] = S_0
        S2_buf = torch.zeros((batch_size, self.N_euler + 1, self.dim)).to(device)
        S2_buf[:, 0, :] = self.L(0.0, V_0, device)
        u_buf = torch.zeros((batch_size, self.N_euler + 1, 2 * self.dim)).to(device)
        # Initialize integral of V
        V_int = torch.zeros_like(V_0).to(device)

        with torch.no_grad():
            # Euler scheme
            with tqdm.tqdm(
                total=self.N_euler,
                position=0,
                leave=True,
                desc="Sampling trajectories",
                disable=disable_tqdm,
            ) as progress_bar:
                for k in range(self.N_euler):
                    S1 = S1_buf[:, k, :]
                    S2 = S2_buf[:, k, :]
                    V = V_buf[:, k, :]
                    # Compute control at t_k
                    if self.multiple_controls:
                        # Compute u_k(log(S^1_{t_k}), V_{t_k}, u_{t_{k-1}})
                        u = self.control[k](
                            torch.concat((torch.log(S1_buf), V_buf, u_buf), dim=1)
                        )
                    else:
                        # Compute u(t_k, u_k(log(S^1_{t_k}), V_{t_k}, u_{t_{k-1}})
                        u = self.control(
                            torch.concat(
                                (
                                    k * self.h * torch.ones((batch_size, 1)).to(device),
                                    torch.log(S1_buf),
                                    V_buf,
                                    u_buf,
                                ),
                                dim=1,
                            )
                        )

                    # Compute gaussian increments
                    B = torch.randn_like(S1)
                    W = self.rho * B + math.sqrt(
                        1 - self.rho * self.rho
                    ) * torch.randn_like(S1)

                    # Update buffers
                    S1_buf[:, k + 1, :] = torch.clamp(
                        S1 + math.sqrt(self.h) * V.sqrt() * S1 * B,
                        min=self.epsilon,
                        max=100.0 / self.epsilon,
                    )
                    V_buf[:, k + 1, :] = torch.nn.functional.relu(
                        V
                        + self.h * self.a * (self.b - V)
                        + self.sigma * math.sqrt(self.h) * V.sqrt() * W
                    )
                    V_int = V_int + self.h * V_buf[:, k + 1, :]
                    S2_buf[:, k + 1, :] = V_int + self.L(
                        (k + 1) * self.h, V_buf[:, k + 1, :]
                    )
                    u_buf[:, k + 1, :] = u

        return S1_buf, S2_buf, V_buf, u_buf

    def objective(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        disable_tqdm: bool = True,
    ) -> torch.Tensor:
        """Computes the control objective J
        Args:
            - batch_size: number of trajectories to sample
            - S_0: initial value of the spot
            - V_0: initial value of the variance
            - device: torch device
            - disable_tqdm: whether to disable the progress bar or not
        Returns:
            - torch.Tensor: tensor J"""
        # Batchify initial conditions
        S_0 = self.S_0.to(device) * torch.ones((batch_size, self.dim)).to(device)
        V_0 = self.V_0.to(device) * torch.ones((batch_size, self.dim)).to(device)
        # Initialize buffers
        V_buf = V_0
        S1_buf = S_0
        S2_buf = self.L(0.0, V_0, device)
        u_buf = torch.zeros((batch_size, 2 * self.dim)).to(device)
        # Initialize integral of V
        V_int = torch.zeros_like(V_0).to(device)
        # Initialize objective terms
        transaction_cost = torch.zeros((batch_size, 1)).to(device)
        benefits = torch.zeros((batch_size, 1)).to(device)

        # Euler scheme
        with tqdm.tqdm(
            total=self.N_euler,
            position=0,
            leave=True,
            desc="Sampling trajectories",
            disable=disable_tqdm,
        ) as progress_bar:
            for k in range(self.N_euler):
                # Compute control at t_k
                if self.multiple_controls:
                    # Compute u_k(log(S^1_{t_k}), V_{t_k}, u_{t_{k-1}})
                    u = self.control[k](
                        torch.concat((torch.log(S1_buf), V_buf, u_buf), dim=1)
                    )
                else:
                    # Compute u(t_k, u_k(log(S^1_{t_k}), V_{t_k}, u_{t_{k-1}})
                    u = self.control(
                        torch.concat(
                            (
                                k * self.h * torch.ones((batch_size, 1)).to(device),
                                torch.log(S1_buf),
                                V_buf,
                                u_buf,
                            ),
                            dim=1,
                        )
                    )

                # Increment transaction cost
                transaction_cost = transaction_cost + torch.sum(
                    self.T_COST
                    * torch.abs(u - u_buf)
                    * torch.concat((S1_buf, S2_buf), dim=1),
                    dim=1,
                    keepdim=True,
                )

                # Compute gaussian increments
                B = torch.randn_like(S1_buf)
                W = self.rho * B + torch.sqrt(
                    1 - torch.square(self.rho)
                ) * torch.randn_like(S1_buf)

                # Update the stochastic processes
                S1 = torch.clamp(
                    S1_buf + math.sqrt(self.h) * V_buf.sqrt() * S1_buf * B,
                    min=self.epsilon,
                    max=100.0 / self.epsilon,
                )
                V = torch.nn.functional.relu(
                    V_buf
                    + self.h * self.a * (self.b - V_buf)
                    + self.sigma * math.sqrt(self.h) * V_buf.sqrt() * W
                )
                V_int = V_int + self.h * V
                S2 = V_int + self.L((k + 1) * self.h, V)

                # Increment benefits
                benefits = benefits + torch.sum(
                    u * torch.concat((S1 - S1_buf, S2 - S2_buf), dim=1),
                    dim=1,
                    keepdim=True,
                )

                # Update buffers
                S1_buf = S1
                S2_buf = S2
                V_buf = V
                u_buf = u

                progress_bar.update(1)

        # Compute last term of transaction cost
        transaction_cost = transaction_cost + torch.sum(
            self.T_COST * torch.abs(u_buf) * torch.concat((S1_buf, S2_buf), dim=1),
            dim=1,
            keepdim=True,
        )

        # Compute w
        w = self.w(torch.ones((batch_size, 1)))

        # Compute payoff
        Z = torch.sum(torch.nn.functional.relu(S1_buf - self.K), dim=1, keepdim=True)

        # Compute objective
        J = w + self.ell(Z - benefits + transaction_cost - w)

        return J
