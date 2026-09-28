"""Training loops: single-domain Adam/L-BFGS, causal windowed training, checkpoints.

Moved out of :mod:`pinn.train` so that module is orchestration only; ``train``,
``get_device``, ``set_seed``, ``make_grid``, ``predict`` and ``residual_at`` are
re-exported there so ``from pinn.train import ...`` keeps working.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import Tensor

from ..config import Config, reference_at, reference_trajectory
from ..history import TrainHistory, SnapshotWriter, flat_grads, flat_params
from ..pinn import PINN, WindowedPINN, build_model, loss_terms, pinn_loss, residual_of as residual, residual_parts
from .collocation import latin_hypercube_points, uniform_points
from .losses import causal_loss
from .optimizers import adam_with_decay, run_lbfgs


def _save_progress(cfg: Config, model, history: TrainHistory, **state) -> None:
    """One atomic, complete resume point; per-window files remain export artifacts."""
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    target = cfg.ckpt_path / "progress.pt"
    pending = target.with_suffix(".tmp")
    torch.save({"config": cfg.record(), "model": model.state_dict(), "history": history,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                **state}, pending)
    pending.replace(target)


def _load_progress(cfg: Config, model, device):
    path = cfg.ckpt_path / "progress.pt"
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6
        state = torch.load(path, map_location=device)
    if json.loads(json.dumps(state["config"])) != json.loads(json.dumps(cfg.record())):
        raise ValueError(f"checkpoint configuration differs from {path}")
    model.load_state_dict(state["model"])
    torch.set_rng_state(state["torch_rng"].cpu())
    np.random.set_state(state["numpy_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    history = state["history"]
    if history.snapshots is not None:
        if history.snapshots.dir.resolve() != (cfg.results_path / "breakdown").resolve():
            raise ValueError("checkpoint snapshot directory differs from run directory")
        kept = {f"epoch_{epoch:06d}.csv" for epoch in history.snapshots.epochs}
        for file in history.snapshots.dir.glob("epoch_*.csv"):
            if file.name not in kept:
                file.unlink()
    return state, history


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_grid(cfg: Config, device: torch.device) -> Tensor:
    t = (uniform_points(cfg.t_span, cfg.n_collocation, cfg.n_windows) if cfg.collocation == "uniform"
         else latin_hypercube_points(cfg.t_span, cfg.n_collocation, cfg.seed))
    return torch.as_tensor(t, dtype=cfg.torch_dtype, device=device)


def train(cfg: Config) -> tuple[PINN, TrainHistory]:
    if cfg.n_windows > 1 or cfg.causal_eps_schedule or cfg.checkpoint_every or cfg.snapshot_every:
        cfg.ensure_record()
    set_seed(cfg.seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg).to(device=device, dtype=cfg.torch_dtype)
    grid = make_grid(cfg, device)
    history = TrainHistory()
    t_ref, ys_ref = reference_trajectory(cfg, n=cfg.n_eval)

    if cfg.snapshot_every > 0:
        grid_t = grid.detach().cpu().numpy().reshape(-1)
        history.snapshots = SnapshotWriter(
            cfg.results_path / "breakdown", grid_t, reference_at(cfg, grid_t)
        )

    progress = (_load_progress(cfg, model, device)
                if cfg.n_windows > 1 or cfg.causal_eps_schedule or cfg.checkpoint_every else None)
    if progress is not None:
        state, history = progress
        if state["complete"]:
            return model, history

    if cfg.print_every:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[setup] device={device.type} | seed={cfg.seed} | {cfg.label}", flush=True)
        print(f"[setup] {n_params} params | {cfg.n_collocation} collocation pts on "
              f"t in [{cfg.t_span[0]}, {cfg.t_span[1]}] | ref traj {len(t_ref)} pts", flush=True)
        schedule = (f"StepLR factor {cfg.lr_decay:g} every {cfg.lr_decay_every} steps"
                    if cfg.lr_decay is not None else f"linear toward {cfg.lr_end:.1e}")
        if cfg.causal_eps_schedule:
            print(f"[setup] {cfg.n_windows} windows, up to {cfg.causal_max_iters} Adam steps per "
                  f"epsilon stage, then up to {cfg.lbfgs_iters} L-BFGS iterations per window; "
                  f"lr {cfg.lr_start:.1e}, {schedule}", flush=True)
        else:
            print(f"[setup] {cfg.epochs} Adam steps per window + up to {cfg.lbfgs_iters} "
                  f"L-BFGS iterations; lr {cfg.lr_start:.1e}, {schedule}", flush=True)
        if history.snapshots is not None:
            print(f"[setup] per-point snapshots every {cfg.snapshot_every} logged evaluations "
                  f"and at final window states -> {history.snapshots.dir}", flush=True)
    t_start = time.perf_counter() - history.wall_clock_s
    if cfg.n_windows > 1 or cfg.causal_eps_schedule:
        if progress is not None and state["mode"] != "windows":
            raise ValueError("checkpoint mode differs from windowed configuration")
        train_windows(model, grid, cfg, history, t_start, t_ref, ys_ref,
                      state["next_window"] if progress is not None else 0)
        return model, history

    adam, sched = adam_with_decay(model.parameters(), cfg)
    start = 0
    if progress is not None:
        if state["mode"] != "single":
            raise ValueError("checkpoint mode differs from single-domain configuration")
        adam.load_state_dict(state["adam"])
        sched.load_state_dict(state["sched"])
        start = state["next_epoch"]
        history.resumed_from = start

    for epoch in range(start, cfg.epochs):
        last = epoch == cfg.epochs - 1
        logging = epoch % cfg.log_every == 0 or last

        adam.zero_grad()
        res, ic, parts = loss_terms(model, grid.clone().requires_grad_(True))
        loss = res + model.gamma * ic if (cfg.ic == "soft" or cfg.end_state is not None) else res
        loss.backward()

        # Snapshot before the step, so the recorded residuals are exactly the ones
        # this epoch's loss and gradient were computed from. Costs one CSV write;
        # the tensors are already in hand.
        if history.snapshots is not None and (epoch % cfg.snapshot_every == 0 or last):
            history.snapshots.write(epoch, parts)
            history.param_epochs.append(epoch)
            history.param_trail.append(flat_params(model).cpu().numpy())
            history.grad_trail.append(flat_grads(model).cpu().numpy())

        lr = adam.param_groups[0]["lr"]
        before = flat_params(model) if logging else None
        adam.step()
        sched.step()
        history.loss.append(loss.item())
        history.loss_phase.append("adam")
        history.loss_window.append(0)

        if logging:
            history.record_step(epoch, model=model, loss=loss, residual=res, ic=ic,
                                lr=lr, params_before=before)
        if epoch % cfg.eval_every == 0 or last:
            history.record_reference(epoch, float(np.mean((predict(model, t_ref) - ys_ref) ** 2)))

        if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
            history.wall_clock_s = time.perf_counter() - t_start
            _save_progress(cfg, model, history, mode="single", complete=False,
                           next_epoch=epoch + 1, adam=adam.state_dict(), sched=sched.state_dict())
        if cfg.print_every and (epoch % cfg.print_every == 0 or last):
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (epoch + 1) * (cfg.epochs - epoch - 1)
            print(f"[adam]  epoch {epoch + 1:>6}/{cfg.epochs} | loss {history.loss[-1]:.4e} | "
                  f"res {res.item():.4e} | ic {ic.item():.4e} | "
                  f"|grad| {history.grad_norm[-1]:.3e} | step {history.update_norm[-1]:.3e} | "
                  f"lr {lr:.2e} | ref_mse {history.ref_mse[-1]:.4e} | "
                  f"{elapsed:6.1f}s elapsed, ~{eta:5.0f}s left", flush=True)

    history.adam_iters = cfg.epochs
    history.wall_clock_s = time.perf_counter() - t_start
    if cfg.print_every:
        print(f"[adam]  done in {time.perf_counter() - t_start:.1f}s | "
              f"final loss {history.loss[-1]:.4e} | best loss {min(history.loss):.4e}", flush=True)

    if cfg.lbfgs_iters > 0:
        def closure() -> Tensor:
            loss = pinn_loss(model, grid.clone().requires_grad_(True))
            history.loss.append(loss.item())
            history.loss_phase.append("lbfgs")
            history.loss_window.append(0)
            n = len(history.loss) - history.adam_iters
            if cfg.print_every and n % 10 == 1:
                print(f"[lbfgs] eval {n:>5} | loss {history.loss[-1]:.4e} | "
                      f"{time.perf_counter() - t_start:6.1f}s elapsed", flush=True)
            return loss

        run_lbfgs(model.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
        history.wall_clock_s = time.perf_counter() - t_start
        if cfg.print_every:
            print(f"[lbfgs] done after {len(history.loss) - history.adam_iters} evals | "
                  f"final loss {history.loss[-1]:.4e}", flush=True)

    if history.snapshots is not None:
        model.zero_grad(set_to_none=True)
        final_parts = residual_parts(model, grid.clone().requires_grad_(True))
        history.snapshots.write(len(history.loss), final_parts)
        history.param_epochs.append(len(history.loss))
        history.param_trail.append(flat_params(model).cpu().numpy())
        final_parts.r.pow(2).mean().backward()
        history.grad_trail.append(flat_grads(model).cpu().numpy())
    history.wall_clock_s = time.perf_counter() - t_start
    if cfg.checkpoint_every:
        _save_progress(cfg, model, history, mode="single", complete=True,
                       next_epoch=cfg.epochs, adam=adam.state_dict(), sched=sched.state_dict())

    return model, history


def train_windows(model, grid: Tensor, cfg: Config, history: TrainHistory, t_start: float,
                  t_ref: np.ndarray, ys_ref: np.ndarray, start_window: int = 0) -> None:
    """Window by window: Adam under each eps until min_i w_i > delta, then L-BFGS, then hand
    the end state to the next window. One window with an empty schedule is plain Adam."""
    windows = list(model.windows) if isinstance(model, WindowedPINN) else [model]
    stages = list(cfg.causal_eps_schedule) or [None]
    cap = cfg.causal_max_iters if cfg.causal_eps_schedule else cfg.epochs
    device = grid.device
    if isinstance(model, WindowedPINN):
        counts = torch.bincount(model.window_of(grid), minlength=len(windows))
        if torch.any(counts == 0):
            raise ValueError("collocation grid leaves an empty window")

    def snapshot(epoch: int, w, trained_until: float, final: bool = False) -> None:
        if history.snapshots is None or (epoch % cfg.snapshot_every and not final):
            return
        if history.snapshots.epochs and epoch == history.snapshots.epochs[-1]:
            return
        history.snapshots.write(epoch, residual_parts(model, grid.clone().requires_grad_(True)),
                                trained_until=trained_until)
        history.param_epochs.append(epoch)
        history.param_trail.append(flat_params(model).cpu().numpy())
        history.grad_trail.append(flat_grads(model).cpu().numpy())
        if w is not None:
            history.weight_profiles.append(w.cpu().numpy())

    for k in range(start_window, len(windows)):
        sub = windows[k]
        edge = np.asarray(sub.tf, dtype=np.float64 if cfg.dtype == "float64" else np.float32)
        coverage = sub.tf if k + 1 == len(windows) else float(
            np.nextafter(edge, np.asarray(-np.inf, dtype=edge.dtype)))
        saved = cfg.ckpt_path / f"window_{k:02d}.pt"
        pts = grid[model.window_of(grid) == k] if isinstance(model, WindowedPINN) else grid
        if isinstance(model, WindowedPINN) and k + 1 < len(windows):
            # The shared edge belongs to the next bucket; also enforce the
            # outgoing window's residual at the state it hands forward.
            endpoint = torch.tensor([[model.edges[k + 1]]], dtype=grid.dtype, device=device)
            pts = torch.cat((pts, endpoint))
        if k and cfg.warm_start:
            with torch.no_grad():
                for p, q in zip(sub.net.parameters(), windows[k - 1].net.parameters()):
                    p.copy_(q)
        adam, sched = adam_with_decay(sub.parameters(), cfg, cap * len(stages))
        for eps in stages:
            stage_start = history.adam_iters
            met = False
            for step in range(cap):
                epoch = len(history.loss)
                model.zero_grad(set_to_none=True)
                t = pts.clone().requires_grad_(True)
                raw_res, ic, parts = loss_terms(sub, t)
                res, w = causal_loss(parts.r, t, eps) if eps is not None else (raw_res, None)
                if w is not None:
                    history.min_w.append(float(w.min()))
                loss = res + sub.gamma * ic if (cfg.ic == "soft" or sub.end_state is not None) else res
                loss.backward()
                snapshot(epoch, w, coverage)
                logging = epoch % cfg.log_every == 0
                before = flat_params(sub) if logging else None
                lr = adam.param_groups[0]["lr"]
                adam.step()
                sched.step()
                history.loss.append(loss.item())
                history.loss_phase.append("adam")
                history.loss_window.append(k)
                history.adam_iters += 1
                if logging:
                    history.record_step(epoch, model=sub, loss=loss, residual=raw_res, ic=ic,
                                        lr=lr, params_before=before)
                if epoch % cfg.eval_every == 0:
                    valid = t_ref <= coverage
                    history.record_reference(epoch, float(np.mean((predict(model, t_ref[valid]) - ys_ref[valid]) ** 2)),
                                             coverage)
                if cfg.print_every and epoch % cfg.print_every == 0:
                    min_w = history.min_w[-1] if w is not None else 1.0
                    print(f"[adam]  window {k:>2} eps {eps} | epoch {epoch:>7} | loss {loss.item():.4e} "
                          f"| min_w {min_w:.3f} | {time.perf_counter() - t_start:7.1f}s", flush=True)
                if w is not None and (history.min_w[-1] > cfg.causal_delta or step == cap - 1):
                    # Certify the updated parameters, which enter the next stage.
                    post = residual_parts(sub, pts.clone().requires_grad_(True))
                    _, post_w = causal_loss(post.r, pts, eps)
                    history.min_w[-1] = float(post_w.min())
                    if history.min_w[-1] > cfg.causal_delta:
                        met = True
                        break
            if eps is not None:
                history.eps_marks.append((len(history.loss), eps))
                history.stage_status.append({"window": k, "eps": eps,
                                             "adam_steps": history.adam_iters - stage_start,
                                             "min_w_final": history.min_w[-1] if history.adam_iters > stage_start
                                                            else float("nan"),
                                             "threshold_met": met})
        if cfg.lbfgs_iters > 0:
            def closure() -> Tensor:
                loss = pinn_loss(sub, pts.clone().requires_grad_(True))
                history.loss.append(loss.item())
                history.loss_phase.append("lbfgs")
                history.loss_window.append(k)
                return loss
            run_lbfgs(sub.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
        cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(sub.state_dict(), saved)
        model.zero_grad(set_to_none=True)
        pinn_loss(sub, pts.clone().requires_grad_(True)).backward()
        done = torch.ones(len(pts), dtype=grid.dtype) if stages[0] is not None else None
        snapshot(len(history.loss), done, coverage, final=True)
        history.window_marks.append(len(history.loss))
        if isinstance(model, WindowedPINN) and k + 1 < len(windows):
            with torch.no_grad():
                end = torch.tensor([[model.edges[k + 1]]], dtype=grid.dtype, device=device)
                model.set_window_start(k + 1, sub(end)[0])
        history.wall_clock_s = time.perf_counter() - t_start
        _save_progress(cfg, model, history, mode="windows", complete=k + 1 == len(windows),
                       next_window=k + 1)
        if cfg.print_every and history.loss:
            print(f"[window] {k:>2} done | loss {history.loss[-1]:.4e} | "
                  f"{time.perf_counter() - t_start:7.1f}s", flush=True)
    history.wall_clock_s = time.perf_counter() - t_start


def predict(model: PINN, t: np.ndarray) -> np.ndarray:
    p = next(model.parameters())
    tt = torch.as_tensor(t, dtype=p.dtype, device=p.device).reshape(-1, 1)
    with torch.no_grad():
        return model(tt).cpu().numpy().astype(np.float64)


def residual_at(model: PINN, t: np.ndarray) -> np.ndarray:
    """ODE residual r(t) = du/dt - f(u) evaluated at arbitrary times."""
    p = next(model.parameters())
    tt = torch.as_tensor(t, dtype=p.dtype, device=p.device).reshape(-1, 1).requires_grad_(True)
    return residual(model, tt).detach().cpu().numpy().astype(np.float64)
