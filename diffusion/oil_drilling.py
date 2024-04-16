from .base import DeepControledDiffusion
import torch
import math
import tqdm
from typing import Callable
from training.network import MLP


class OilDrilling(DeepControledDiffusion):
    """This class implements the oil drilling problem"""

    def __init__(
        self,
        T: float = 1.0,
        N_euler: int = 50,
        dim: int = 1,
        mu: float = 0.01,
        sigma: float = 0.2,
        rho: float = 0.01,
        epsilon: float = 0.0,
        xi_s: float = 0.005,
        K0: float = 5.0,
        xi_e: float = 0.01,
        qS: float = 10.0,
        P0: torch.Tensor = 1.0 * torch.ones(1),
        U: Callable = (lambda x: x),
    ) -> None:
        super().__init__(T, N_euler, dim)
        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.epsilon = epsilon
        self.xi_s = xi_s
        self.K0 = K0
        self.xi_e = xi_e
        self.qS = qS
        self.P0 = P0
        self.U = U
        self.c_s = lambda x: torch.exp(self.xi_s * x) - 1.0
        self.c_e = lambda x: torch.exp(self.xi_e * x)
        self.name = "Oil Drilling"

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

    def train_mode(self) -> None:
        """Sets the control to train mode"""
        if not self.multiple_controls:
            self.control.train()
        else:
            for control in self.control:
                control.train()

    def eval_mode(self) -> None:
        """Sets the control to eval mode"""
        if not self.multiple_controls:
            self.control.eval()
        else:
            for control in self.control:
                control.eval()

    def apply_constraints(self, q_t: torch.Tensor, S_t: torch.Tensor) -> torch.Tensor:
        """Helper function that casts the contraints on the control"""
        qv_t, qs_t, qvs_t = q_t[:, 0:1], q_t[:, 1:2], q_t[:, 2:3]
        qv_t = self.K0 - torch.nn.functional.relu(self.K0 - qv_t)
        qs_t = self.K0 - qv_t - torch.nn.functional.relu(self.K0 - qv_t - qs_t)
        qvs_t = self.qS - torch.nn.functional.relu(self.qS - qvs_t)
        qvs_t = S_t / self.h - torch.nn.functional.relu(S_t / self.h - qvs_t)
        return torch.concat((qv_t, qs_t, qvs_t), dim=1)

    def sample_traj(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        disable_tqdm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples trajectory along with the corresponding control
        Args:
            - batch_size: number of trajectories to sample
            - device: torch device
            - disable_tqdm: whether to disable the progress bar or not
        Returns:
            - tuple[torch.Tensor, torch.Tensor]: tuple containing trajectory of X and corresponding control u
        """
        with torch.no_grad():
            # Initialize buffers
            P_buf = torch.zeros((batch_size, self.N_euler + 1, self.dim)).to(device)
            P_buf[:, 0, :] = self.P0.to(device) * torch.ones((batch_size, self.dim)).to(
                device
            )
            S_buf = torch.zeros((batch_size, self.N_euler + 1, self.dim)).to(device)
            E_buf = torch.zeros((batch_size, self.N_euler + 1, self.dim)).to(device)
            q_buf = torch.zeros((batch_size, self.N_euler, 3 * self.dim)).to(device)

            # Euler scheme
            with tqdm.tqdm(
                total=self.N_euler,
                position=0,
                leave=True,
                desc="Sampling trajectories",
                disable=disable_tqdm,
            ) as progress_bar:
                for k in range(self.N_euler):
                    P_t = P_buf[:, k, :]
                    E_t = E_buf[:, k, :]
                    S_t = S_buf[:, k, :]

                    # Compute control
                    if self.multiple_controls:
                        # Compute u_k(X_{t_k})
                        q_t = self.control[k](torch.concat((P_t, E_t, S_t), dim=1))
                    else:
                        # Compute u(t_k, X_{t_{k}})
                        q_t = self.control(
                            torch.concat(
                                (
                                    k * self.h * torch.ones((batch_size, 1)).to(device),
                                    P_t,
                                    E_t,
                                    S_t,
                                ),
                                dim=1,
                            )
                        )
                    # Cast control on the right space
                    q_t = self.apply_constraints(q_t, S_t)
                    qv_t, qs_t, qvs_t = q_t[:, 0:1], q_t[:, 1:2], q_t[:, 2:3]
                    q_buf[:, k, :] = q_t

                    # Update buffers
                    E_buf[:, k + 1, :] = E_t + self.h * (qv_t + qs_t)
                    S_buf[:, k + 1, :] = S_t + self.h * (qs_t - qvs_t)
                    P_buf[:, k + 1, :] = P_t * torch.exp(
                        (self.mu - 0.5 * self.sigma**2) * self.h
                        + self.sigma
                        * math.sqrt(self.h)
                        * torch.randn_like(P_t).to(device)
                    )

                    # Update progress bar
                    progress_bar.update(1)

            return P_buf, S_buf, E_buf, q_buf

    def objective(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        disable_tqdm: bool = True,
    ) -> torch.Tensor:
        """Computes the control objective J
        Args:
            - batch_size: number of trajectories to sample
            - device: torch device
            - disable_tqdm: whether to disable the progress bar or not
        Returns:
            - torch.Tensor: tensor J"""
        # Initialize buffers
        P_t = self.P0.to(device) * torch.ones((batch_size, self.dim)).to(device)
        E_t = torch.zeros((batch_size, self.dim)).to(device)
        S_t = torch.zeros((batch_size, self.dim)).to(device)
        J = torch.zeros((batch_size, 1)).to(device)

        # Euler scheme
        with tqdm.tqdm(
            total=self.N_euler,
            position=0,
            leave=True,
            desc="Sampling trajectories",
            disable=disable_tqdm,
        ) as progress_bar:
            for k in range(self.N_euler):
                # Compute control
                if self.multiple_controls:
                    # Compute u_k(X_{t_k})
                    q_t = self.control[k](torch.concat((P_t, E_t, S_t), dim=1))
                else:
                    # Compute u(t_k, X_{t_{k}})
                    q_t = self.control(
                        torch.concat(
                            (
                                k * self.h * torch.ones((batch_size, 1)).to(device),
                                P_t,
                                E_t,
                                S_t,
                            ),
                            dim=1,
                        )
                    )
                # Cast control on the right space
                q_t = self.apply_constraints(q_t, S_t)
                qv_t, qs_t, qvs_t = q_t[:, 0:1], q_t[:, 1:2], q_t[:, 2:3]

                # Compute the objective terms
                benefits = P_t * (qv_t + (1.0 - self.epsilon) * qvs_t)
                extraction_cost = (qv_t + qs_t) * self.c_e(E_t)
                storage_cost = self.c_s(S_t)

                # Update buffers
                E_t = E_t + self.h * (qv_t + qs_t)
                S_t = S_t + self.h * (qs_t - qvs_t)
                J = J - self.h * math.exp(-self.rho * k * self.h) * self.U(
                    benefits - extraction_cost - storage_cost
                )
                P_t = P_t * torch.exp(
                    (self.mu - 0.5 * self.sigma**2) * self.h
                    + self.sigma * math.sqrt(self.h) * torch.randn_like(P_t).to(device)
                )

                # Update progress bar
                progress_bar.update(1)

        # Compute last discretization step where we dump all the oil
        J = J - math.exp(-self.rho * self.T) * self.U((1.0 - self.epsilon) * P_t * S_t)

        return J
