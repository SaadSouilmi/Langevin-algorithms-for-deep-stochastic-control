import torch.optim as optim


class PiecewiseConstantScheduler:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        target_lr: float,
        target_sigma: float,
        total_iters: int,
    ) -> None:
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.target_sigma = target_sigma
        self.total_iters = total_iters
        self.cur_epoch = 0

    def step(
        self,
    ) -> None:
        if self.cur_epoch == self.total_iters - 1:
            for group in self.optimizer.param_groups:
                group["lr"] = self.target_lr
                if "sigma" in group:
                    group["sigma"] = self.target_sigma
        self.cur_epoch += 1

    def get_last_lr(self) -> list[float]:
        lr = []
        for group in self.optimizer.param_groups:
            lr.append(group["lr"])

        return lr

    def get_last_sigma(self) -> list[float]:
        sigma = []
        for group in self.optimizer.param_groups:
            if "sigma" in group:
                sigma.append(group["sigma"])

        if sigma:
            return sigma
        else:
            return [0]
