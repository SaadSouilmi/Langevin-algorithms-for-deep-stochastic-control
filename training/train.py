import torch
import math
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from diffusion.base import DeepControledDiffusion
from typing import Union


def train(
    model: DeepControledDiffusion,
    optimizer: Union[torch.optim.Optimizer, list[torch.optim.Optimizer]],
    scheduler: Union[
        torch.optim.lr_scheduler.LRScheduler, list[torch.optim.lr_scheduler.LRScheduler]
    ],
    optimizer_name: str,
    epochs: int,
    train_size: int,
    test_size: int,
    train_batch: int,
    test_batch: int,
    disable_tqdm: bool = False,
) -> tuple[list, list, list]:
    """This function implements the training loop
    Args:
        - model: DeepControledDiffusion object that implements an SOC problem
        - optimizer: either a torch optimizer or a list of optimizers
        - scheduler: either a torch scheduler or a list of schedulers
        - optimizer_name: name of the optimization algorithm
        - epochs: number of training epochs
        - train_size: number of batches to run during training
        - test_size: number of batches to run during evaluation
        - train_batch: batch size during train
        - test_batch: batch size during test
        - disable_tqdm: whether or not to show a progress bar
    Returns:
        - tuple[list, list, list]: train and test losses along with the test confidence interval radius
    """
    train_losses = []
    test_losses = []
    test_ci_radius = []

    with tqdm.tqdm(
        total=epochs,
        position=0,
        leave=True,
        desc=f"Training {model.name} with {optimizer_name}",
        disable=disable_tqdm,
    ) as progress_bar:
        for epoch in range(epochs):
            running_train_loss = 0.0
            running_test_loss = 0.0
            running_test_var = 0.0

            model.train_mode()
            for batch in range(train_size):
                if not isinstance(optimizer, list):
                    optimizer.zero_grad()
                else:
                    for opt in optimizer:
                        opt.zero_grad()
                J = model.objective(train_batch)
                loss = J.mean()
                loss.backward()
                if not isinstance(optimizer, list):
                    optimizer.step()
                else:
                    for opt in optimizer:
                        opt.step()

                running_train_loss += loss.item()

            train_losses.append(running_train_loss / train_size)
            if not isinstance(scheduler, list):
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
                sigma = scheduler.get_last_sigma()
            else:
                for sched in scheduler:
                    sched.step()
                lr = sched.get_last_lr()[0]
                sigma = sched.get_last_sigma()

            model.eval_mode()
            with torch.no_grad():
                for batch in range(test_size):
                    J = model.objective(test_batch)

                    running_test_loss += J.mean().item()
                    running_test_var += J.var().item()

                test_losses.append(running_test_loss / test_size)
                test_ci_radius.append(
                    1.96
                    * math.sqrt(
                        running_test_var
                        * (test_batch - 1)
                        / (test_size * test_batch - 1)
                    )
                    / math.sqrt(test_size * test_batch)
                )

            progress_bar.set_description(
                f"{model.name}: Epoch {epoch}, {optimizer_name}, lr={lr:.3f}, sigma={sigma[0]:.3f}, train={running_train_loss/train_size:.3f}, test={running_test_loss/test_size:.3f}"
            )
            progress_bar.update(1)

    return train_losses, test_losses, test_ci_radius


def plot_langevin_loss(
    test_loss: list,
    test_ci: list,
    test_loss_langevin: list,
    test_ci_langevin: list,
    test_loss_llangevin: Union[list, list[list]] = None,
    test_ci_llangevin: Union[list, list[list]] = None,
    ll: Union[int, list[int]] = 10,
    name: str = "SGD",
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    save_fig: str = None,
    figsize: tuple[int, int] = (7, 5),
) -> None:
    """Helper function that displays the test loss for an algorithm vs it's Langevin version
    Args:
        - test_loss: test loss of the vanilla algo
        - test_ci: ci radius of the vanilla algo
        - test_loss_langevin: test loss of the Langevin algo
        - test_ci_langevin: ci radius of the Langevin algo
        - test_loss_llangevin: layer langevin test losses
        - test_ci_llangevin: layer langevin ci radiuses
        - ll: values of layer langevin used
        - name: name of the algo
        - xlim: limits of the x axis
        - ylim: limits of the y axis
        - save_fig: path for saving the graph
        - figsize: figsize
    """
    fig = plt.figure(figsize=figsize)
    plt.plot(
        np.arange(len(test_loss)), test_loss, marker="o", mec="k", ms=3, label=name
    )
    plt.fill_between(
        np.arange(len(test_loss)),
        np.array(test_loss) - np.array(test_ci),
        np.array(test_loss) + np.array(test_ci),
        alpha=0.2,
    )
    plt.plot(
        np.arange(len(test_loss_langevin)),
        test_loss_langevin,
        marker="o",
        mec="k",
        ms=3,
        label=f"L{name}",
    )
    plt.fill_between(
        np.arange(len(test_loss_langevin)),
        np.array(test_loss_langevin) - np.array(test_ci_langevin),
        np.array(test_loss_langevin) + np.array(test_ci_langevin),
        alpha=0.2,
    )
    if test_loss_llangevin:
        if isinstance(test_loss_llangevin[0], list):
            if not isinstance(test_ci_llangevin[0], list) or not isinstance(ll, list):
                raise ValueError("Expected a list of test_ci_llangevin and ll")
            for test_loss_ll, test_ci_ll, ll_rate in zip(
                test_loss_llangevin, test_ci_llangevin, ll
            ):
                plt.plot(
                    np.arange(len(test_loss_ll)),
                    test_loss_ll,
                    marker="o",
                    mec="k",
                    ms=3,
                    label=f"LL{name}-{ll_rate}%",
                )
                plt.fill_between(
                    np.arange(len(test_loss_ll)),
                    np.array(test_loss_ll) - np.array(test_ci_ll),
                    np.array(test_loss_ll) + np.array(test_ci_ll),
                    alpha=0.2,
                )
        else:
            plt.plot(
                np.arange(len(test_loss_llangevin)),
                test_loss_llangevin,
                marker="o",
                mec="k",
                ms=3,
                label=f"LL{name}-{ll}%",
            )
            plt.fill_between(
                np.arange(len(test_loss_llangevin)),
                np.array(test_loss_llangevin) - np.array(test_ci_llangevin),
                np.array(test_loss_llangevin) + np.array(test_ci_llangevin),
                alpha=0.2,
            )

    plt.xlabel("Epochs")
    plt.ylabel(r"$J(u_\theta)$")
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    legend = plt.legend(fancybox=True, edgecolor="k", loc=0)
    legend.get_frame().set_linewidth(0.5)

    if save_fig:
        fig.savefig(save_fig)
    plt.show()
    plt.close()
