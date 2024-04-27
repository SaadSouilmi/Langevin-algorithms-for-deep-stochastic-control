import torch
from torch.optim.optimizer import Optimizer


class LAdagrad(Optimizer):
    """Implements LAdagrad algorithm based on the pytorch implementation.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        sigma=1e-3,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= sigma:
            raise ValueError(f"Invalid noise std: {sigma}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            sigma=sigma,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )
        super(LAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["sum"] = torch.full_like(
                    p.data,
                    initial_accumulator_value,
                    memory_format=torch.preserve_format,
                )

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
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

                grad = p.grad.data
                state = self.state[p]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )
                    grad = grad.add(group["weight_decay"], p.data)

                clr = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])

                if grad.is_sparse:
                    grad = (
                        grad.coalesce()
                    )  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    state["sum"].add_(make_sparse(grad_values.pow(2)))
                    std = state["sum"].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(group["eps"])
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:

                    state["sum"].addcmul_(grad, grad, value=1)
                    std = state["sum"].sqrt().add_(group["eps"])
                    preconditionner = torch.div(1, std)
                    noise_std = group["sigma"] * (preconditionner.mul(clr)).sqrt()
                    noise = p.data.new(p.data.size()).normal_(mean=0, std=1) * noise_std

                    # Perform gradient step
                    p.data.addcmul_(grad, preconditionner, value=-clr)

                    # Perform noise step
                    p.data.add_(noise)

        return loss
