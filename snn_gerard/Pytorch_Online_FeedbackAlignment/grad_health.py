import torch
import wandb


@torch.no_grad()
def log_gradient_health(model, x, targets, step, rel=0.002, n_seqs=8, prefix="graddiag"):
    """
    Tripwire for the e-prop "gradients don't point downhill" bug.

    For each weight group it takes a small step along -g_eprop and checks whether
    the task loss actually decreases:
        dloss < 0  -> healthy (gradient is a descent direction)
        dloss >= 0 -> BROKEN  (gradient does not reduce the loss)

    Cheap (a handful of forward passes, no BPTT). Safe to call anywhere: it
    restores weights, .grad, c_reg and learning_signal_mode exactly, so it never
    perturbs the real optimizer step.
    """
    x, targets = x[:n_seqs], targets[:n_seqs]

    groups = {
        "W_in":  model.hidden_layer.input_weights.weight,
        "W_rec": model.hidden_layer.recurrent_weights.weight,
        "W_out": model.readout.weight,
    }

    saved_w    = {k: w.detach().clone() for k, w in groups.items()}
    saved_grad = {n: (p.grad.detach().clone() if p.grad is not None else None)
                  for n, p in model.named_parameters()}
    saved_creg, saved_mode = model.c_reg, model.learning_signal_mode
    real_log = wandb.log
    wandb.log = lambda *a, **k: None   # silence forward()'s internal per-step logging

    def task_loss():
        out = model(x)                                  # no targets -> no grad written
        return 0.5 * ((out - targets) ** 2).sum(-1).mean().item()

    try:
        # pure TASK gradient: regulariser off, symmetric (true) feedback
        model.c_reg, model.learning_signal_mode = 0.0, "symmetric"
        for p in model.parameters():
            p.grad = None
        model(x, targets=targets, log_step=0)           # e-prop writes .grad in-place
        g = {k: groups[k].grad.detach().clone() for k in groups}

        L0 = task_loss()
        metrics = {}
        for k, w in groups.items():
            step_size = rel * (w.norm() / (g[k].norm() + 1e-12))
            w.add_(-step_size * g[k])                   # W <- W - step * grad
            metrics[f"{prefix}/dloss_{k}"] = (task_loss() - L0) / (L0 + 1e-12)
            w.copy_(saved_w[k])                         # restore exactly
        metrics[f"{prefix}/worst_hidden_dloss"] = max(
            metrics[f"{prefix}/dloss_W_in"], metrics[f"{prefix}/dloss_W_rec"]
        )
        metrics[f"{prefix}/grad_check_step"] = step
    finally:
        wandb.log = real_log
        model.c_reg, model.learning_signal_mode = saved_creg, saved_mode
        for n, p in model.named_parameters():
            p.grad = saved_grad[n]

    wandb.log(metrics)
    return metrics