import torch
from torch.optim.optimizer import Optimizer, required
import math


class LSGD(Optimizer):
    """Implements LSGD based on the pytorch implementation of SGD.
    In this version of the algorithm, the noise is isotropic.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        sigma (float): noise standard deviation
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(
        self,
        params,
        lr=required,
        sigma=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if sigma is not required and sigma < 0.0:
            raise ValueError(f"Invalid noise std: {sigma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, sigma=sigma, weight_decay=weight_decay)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

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
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Compute the gradient
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)  # add gradient of L2 penalty

                # Compute the noise
                noise_std = group["sigma"] * torch.Tensor([group["lr"]]).sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0, std=1) * noise_std

                # Perform gradient step
                p.data.add_(d_p, alpha=-group["lr"])

                # Perform noise step
                p.data.add_(noise)

        return loss


class APLSGD(Optimizer):
    """Implements LSGD based on the pytorch implementation of SGD.
    In this version of the algorithm, the noise is isotropic.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        sigma (float): noise standard deviation
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(
        self,
        params,
        lr=required,
        sigma=required,
        momentum=0.9,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if sigma is not required and sigma < 0.0:
            raise ValueError(f"Invalid noise std: {sigma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum <= 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr, sigma=sigma, momentum=momentum, weight_decay=weight_decay
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(APLSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(APLSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

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
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Compute the gradient
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)  # add gradient of L2 penalty

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    mom = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    var = param_state["var_buffer"] = torch.clone(d_p).detach().zero_()
                else:
                    mom_buf = param_state["momentum_buffer"]
                    mom = mom_buf.mul(momentum).add(d_p, alpha=1 - momentum)
                    var = param_state["var_buffer"]
                    var.mul_(momentum).addcmul_(
                        d_p - mom, d_p - mom_buf, value=1 - momentum
                    )
                    param_state["momentum_buffer"] = mom

                # Compute the noise
                noise_std = group["sigma"] * math.sqrt(group["lr"]) * var.sqrt()
                noise = (
                    p.data.new(p.data.size())
                    .normal_(mean=0, std=1)
                    .mul(noise_std)
                    .add(mom)
                )

                # Perform gradient step
                p.data.add_(d_p, alpha=-group["lr"])

                # Perform noise step
                p.data.add_(noise, alpha=-1)

        return loss
