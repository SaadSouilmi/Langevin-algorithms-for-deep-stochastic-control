import torch
from torch.optim.optimizer import Optimizer


class LAdadelta(Optimizer):
    """Implements LAdadelta algorithm based on the pytorch implementation.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        sigma (float): noise standard deviation
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(
        self, params, lr=1.0, sigma=0.0, betas=(0.95, 0.95), eps=1e-6, weight_decay=0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= sigma:
            raise ValueError(f"Invalid noise std: {sigma}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, sigma=sigma, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super(LAdadelta, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state["acc_delta"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )

                square_avg, acc_delta = state["square_avg"], state["acc_delta"]
                eps = group["eps"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Compute gradient and preconditionner
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adadelta does not support sparse gradients")
                if group["weight_decay"] != 0:
                    grad = grad.add(
                        p.data, alpha=group["weight_decay"]
                    )  # add gradient of L2 penalty

                square_avg.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)
                std = square_avg.add(eps).sqrt()
                preconditionner = acc_delta.add(eps).sqrt().div(std)

                # Compute noise
                noise_std = group["sigma"] * (preconditionner.mul(group["lr"])).sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0, std=1) * noise_std

                # Perform gradient step
                delta = -group["lr"] * grad.mul(preconditionner) + noise
                p.data.add_(delta)

                acc_delta.mul_(beta2).addcmul_(delta, delta, value=1 - beta2)

        return loss
