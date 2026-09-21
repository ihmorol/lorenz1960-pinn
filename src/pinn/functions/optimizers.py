from typing import Callable

import torch


def adam_with_decay(params, cfg):
    adam = torch.optim.Adam(params, lr=cfg.lr_start)
    if cfg.lr_decay is None:
        decay = cfg.lr_end / cfg.lr_start
        sched = torch.optim.lr_scheduler.LambdaLR(
            adam, lambda e: 1.0 + (decay - 1.0) * min(e, cfg.epochs) / cfg.epochs)
    else:
        sched = torch.optim.lr_scheduler.StepLR(adam, cfg.lr_decay_every, cfg.lr_decay)
    return adam, sched


def run_lbfgs(params, closure: Callable[[], torch.Tensor], iters: int, dtype: torch.dtype) -> None:
    # 1e-14 is below float32 resolution, which is why L-BFGS used to stop after a few evals.
    tol = 1e-16 if dtype == torch.float64 else 1e-14
    opt = torch.optim.LBFGS(params, max_iter=iters, history_size=50, tolerance_grad=1e-12,
                            tolerance_change=tol, line_search_fn="strong_wolfe")

    def wrapped():
        opt.zero_grad()
        loss = closure()
        loss.backward()
        return loss

    opt.step(wrapped)
