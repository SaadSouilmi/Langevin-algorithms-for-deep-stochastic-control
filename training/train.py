import torch
import math
import tqdm


def train(
    model,
    optimizer,
    scheduler,
    optimizer_name,
    epochs,
    train_size,
    test_size,
    train_batch,
    test_batch,
    disable_tqdm: bool = False,
):
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
                if not model.multiple_controls:
                    optimizer.zero_grad()
                else:
                    for opt in optimizer:
                        opt.zero_grad()
                J = model.objective(train_batch)
                loss = J.mean()
                loss.backward()
                if not model.multiple_controls:
                    optimizer.step()
                else:
                    for opt in optimizer:
                        opt.step()

                running_train_loss += loss.item()

            train_losses.append(running_train_loss / train_size)
            if not model.multiple_controls:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                for sched in scheduler:
                    sched.step()
                lr = sched.get_last_lr()[0]

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
                f"{model.name}: Epoch {epoch}, {optimizer_name}, lr={lr:.5f}, train={running_train_loss/train_size:.3f}, test={running_test_loss/test_size:.3f}"
            )
            progress_bar.update(1)

    return train_losses, test_losses, test_ci_radius
