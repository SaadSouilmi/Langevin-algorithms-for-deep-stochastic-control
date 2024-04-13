import torch
from torch.optim.optimizer import Optimizer


class LRMSprop(Optimizer):
    """Implements LRMSprop algorithm based on the implementation of pytorch.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        sigma (float)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, sigma=1e-3, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(lr=lr, sigma=sigma, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(LRMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LRMSprop, self).__setstate__(state)

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
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                # Compute gradient and preconditionner
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay']) # add gradient of L2 penalty

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1-alpha)

                avg = square_avg.sqrt().add_(group['eps'])
                preconditionner = torch.div(1, avg)
                
                # Compute noise
                noise_std = group['sigma'] * (preconditionner.mul(group['lr'])).sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0,
                                                          std=1) * noise_std
                
                # Perform gradient step
                p.data.add_(grad.mul(preconditionner), alpha=-group['lr'])

                # Perform noise step
                p.data.add_(noise)

        return loss