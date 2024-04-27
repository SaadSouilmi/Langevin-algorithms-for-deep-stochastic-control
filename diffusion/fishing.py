from .base import DeepControledDiffusion
import torch
import math
import tqdm
from training.network import MLP


class Fishing(DeepControledDiffusion):
    """This class implements the fishing quota problem"""

    def __init__(
        self,
        T: float = 1.0,
        N_euler: int = 50,
        dim: int = 5,
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
        device: torch.device = torch.device("cpu"),
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
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_lbound = init_lbound
        self.init_ubound = init_ubound
        self.device = device
        self.name = "Fishing"

    def set_control(
        self,
        control_config: dict,
        multiple_controls: bool = False,
    ) -> None:
        """Function that sets the control
        Args:
            - control_config: dictionnary containing the config of the control
            - mutiple_controls: boolean determining whether we use mutiple controls or not
        """
        self.multiple_controls = multiple_controls
        if multiple_controls:
            self.control = [
                MLP(**control_config).to(self.device) for k in range(self.N_euler + 1)
            ]
        else:
            self.control = MLP(**control_config).to(self.device)

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

    def sample_start(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """Samples X_0 from a gaussian clipped to [lbound, ubound]
        Args:
            - batch_size: number of initial conditions to sample"""
        # broadcast the mean tensor and cast to device
        batched_mean = self.init_mean * torch.ones(
            (batch_size, len(self.init_mean))
        ).to(self.device)
        return torch.clamp(
            torch.normal(batched_mean, std=self.init_std).to(self.device),
            min=self.init_lbound,
            max=self.init_ubound,
        )

    def sample_traj(
        self,
        batch_size: int = 1,
        disable_tqdm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples trajectory along with the corresponding control
        Args:
            - batch_size: number of trajectories to sample
            - disable_tqdm: whether to disable the progress bar or not
        Returns:
            - tuple[torch.Tensor, torch.Tensor]: tuple containing trajectory of X and corresponding control u
        """

        with torch.no_grad():
            # Initialise buffers for trajectory and control
            X_buf = torch.ones((batch_size, self.N_euler + 1, self.dim)).to(self.device)
            u_buf = torch.ones((batch_size, self.N_euler + 1, self.dim)).to(self.device)

            # Sample X_0
            X_buf[:, 0, :] = self.sample_start(batch_size)

            # Euler Scheme
            with tqdm.tqdm(
                total=self.N_euler,
                position=0,
                leave=True,
                desc=f"Sampling trajectories on {self.device} batch size = {batch_size}",
                disable=disable_tqdm,
            ) as progress_bar:
                for k in range(self.N_euler):
                    # X_tk
                    X = X_buf[:, k, :]
                    # Compute control at X_tk
                    if self.multiple_controls:
                        # Compute u_k(X_tk)
                        u = self.control[k](X)
                    else:
                        # Compute u(t_k, X_tk)
                        u = self.control(
                            torch.concat(
                                (
                                    k
                                    * self.h
                                    * torch.ones((batch_size, 1)).to(self.device),
                                    X,
                                ),
                                dim=1,
                            )
                        )
                    # Cast u to the right interval
                    u = self.u_m + (self.u_M - self.u_m) * u
                    # Compute drift and noise terms
                    drift = self.h * X * (self.r - u - X.matmul(self.kappa_t))
                    noise = (
                        math.sqrt(self.h)
                        * X
                        * torch.normal(torch.zeros_like(X))
                        .to(self.device)
                        .matmul(self.sigma_t)
                    )
                    # Update buffers
                    X_buf[:, k + 1, :] = X + drift + noise
                    u_buf[:, k, :] = u

                    # Update progress bar
                    progress_bar.update(1)

            X = X_buf[:, self.N_euler, :]
            # Compute last control value
            if self.multiple_controls:
                # Compute u_N(X_tN)
                u = self.control[k + 1](X)
            else:
                # Compute u(t_N, X_tN)
                u = self.control(
                    torch.concat(
                        (
                            (k + 1)
                            * self.h
                            * torch.ones((batch_size, 1)).to(self.device),
                            X,
                        ),
                        dim=1,
                    )
                )
            u = self.u_m + (self.u_M - self.u_m) * u
            u_buf[:, k + 1, :] = u

        return X_buf, u_buf

    def objective(
        self,
        batch_size: int = 1,
        disable_tqdm: bool = True,
    ) -> torch.Tensor:
        """Computes the control objective J
        Args:
            - batch_size: number of trajectories to sample
            - disable_tqdm: whether to disable the progress bar or not
        Returns:
            - torch.Tensor: tensor J"""
        # Initialize J
        J = torch.zeros((batch_size, 1)).to(self.device)
        # Initialize X_0
        X_buf = self.sample_start(batch_size)
        # Initialize Control
        if not self.multiple_controls:
            # Compute u(0, X_0)
            u_buf = self.control(
                torch.concat(
                    (torch.zeros((batch_size, 1)).to(self.device), X_buf),
                    dim=1,
                )
            )
        else:
            # Compute u_0(X_0)
            u_buf = self.control[0](X_buf)
        # Cast u to the right interval
        u_buf = self.u_m + (self.u_M - self.u_m) * u_buf

        # Euler scheme
        with tqdm.tqdm(
            total=self.N_euler,
            position=0,
            leave=True,
            desc=f"Sampling trajectories on {self.device} batch size = {batch_size}",
            disable=disable_tqdm,
        ) as progress_bar:
            for k in range(self.N_euler):
                drift = self.h * X_buf * (self.r - u_buf - X_buf.matmul(self.kappa_t))
                noise = (
                    math.sqrt(self.h)
                    * X_buf
                    * torch.normal(torch.zeros_like(X_buf))
                    .to(self.device)
                    .matmul(self.sigma_t)
                )
                X = X_buf + drift + noise

                if self.multiple_controls:
                    # Compute u_{k+1}(X_{t_{k+1}})
                    u = self.control[k + 1](X)
                else:
                    # Compute u(t_{k+1}, X_{t_{k+1}})
                    u = self.control(
                        torch.concat(
                            (
                                (k + 1)
                                * self.h
                                * torch.ones((batch_size, 1)).to(self.device),
                                X,
                            ),
                            dim=1,
                        )
                    )

                # Increment objective
                J = J + self.h * torch.sum(
                    torch.square(X - self.X_d)
                    - self.alpha * u
                    + (self.beta / self.h) * torch.square(u - u_buf),
                    dim=1,
                    keepdim=True,
                )

                # Update buffers
                X_buf = X
                u_buf = u

                # Update progress bar
                progress_bar.update(1)

        return J
