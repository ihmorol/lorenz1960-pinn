"""Training engine: the optimisation loops, plus the runtime a run needs.

Split out of :mod:`pinn.train` so that module reads as pure orchestration
(load, train, save). Two loops live here and nowhere else:

* :class:`Trainer` -- one run, dispatched to a single network (plain Adam,
  optionally L-BFGS) or to a window per time slice under the causal schedule.
* the step bodies -- the arithmetic is unchanged, only relocated, so every
  snapshot, resume point, causal advance test and end-state hand-off keeps its
  original order and therefore its original numbers.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import Tensor

from ..config import Config, reference_at, reference_trajectory
from ..history import SnapshotWriter, TrainHistory, flat_grads, flat_params
from ..pinn import WindowedPINN, build_model, loss_terms, pinn_loss, residual_parts
from .collocation import make_grid
from .losses import causal_loss
from .optimizers import adam_with_decay, run_lbfgs
from .reporting import predict


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class Trainer:
    """Runs one :class:`Config` to a trained model and its telemetry.

    ``resumed`` is the Adam epoch a single-network run picked up from, or ``None``.
    It is recorded because :func:`pinn.train.save_results` writes it out.
    """

    def __init__(self, cfg: Config) -> None:
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = get_device()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.model = build_model(cfg).to(device=self.device, dtype=cfg.torch_dtype)
        self.grid = make_grid(cfg, self.device)
        self.history = TrainHistory()
        self.resumed: int | None = None

    # -- shared pieces ------------------------------------------------------
    def _open_snapshots(self) -> None:
        cfg = self.cfg
        if cfg.snapshot_every <= 0:
            return
        grid_t = self.grid.detach().cpu().numpy().reshape(-1)
        self.history.snapshots = SnapshotWriter(
            cfg.results_path / "breakdown", grid_t, reference_at(cfg, grid_t))

    def _snapshot(self, epoch: int, parts=None, w=None, final: bool = False) -> None:
        """Write one per-point CSV, if due. ``parts`` is reused when already in hand."""
        cfg = self.cfg
        snap = self.history.snapshots
        if snap is None or (epoch % cfg.snapshot_every and not final):
            return
        if parts is None:
            parts = residual_parts(self.model, self.grid.clone().requires_grad_(True))
        snap.write(epoch, parts)
        self.history.param_epochs.append(epoch)
        self.history.param_trail.append(flat_params(self.model).cpu().numpy())
        self.history.grad_trail.append(flat_grads(self.model).cpu().numpy())
        if w is not None:
            self.history.weight_profiles.append(w.cpu().numpy())

    def _announce(self, n_ref: int) -> None:
        cfg = self.cfg
        if not cfg.print_every:
            return
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[setup] device={self.device.type} | seed={cfg.seed} | {cfg.label}", flush=True)
        print(f"[setup] {n_params} params | {cfg.n_collocation} collocation pts on "
              f"t in [{cfg.t_span[0]}, {cfg.t_span[1]}] | ref traj {n_ref} pts", flush=True)
        print(f"[setup] {cfg.epochs} Adam epochs (lr {cfg.lr_start:.1e} -> {cfg.lr_end:.1e}) "
              f"+ {cfg.lbfgs_iters} L-BFGS iters", flush=True)
        if self.history.snapshots is not None:
            n_snap = cfg.epochs // cfg.snapshot_every + 1
            print(f"[setup] per-point snapshots every {cfg.snapshot_every} epochs "
                  f"(~{n_snap} files) -> {self.history.snapshots.dir}", flush=True)

    # -- entry point --------------------------------------------------------
    def run(self) -> tuple[torch.nn.Module, TrainHistory]:
        cfg = self.cfg
        t_ref, ys_ref = reference_trajectory(cfg, n=cfg.n_eval)
        self._open_snapshots()
        self._announce(len(t_ref))
        t0 = time.perf_counter()
        if cfg.n_windows > 1 or cfg.causal_eps_schedule:
            self._run_windows(t0, t_ref, ys_ref)
        else:
            self._run_single(t0, t_ref, ys_ref)
        return self.model, self.history

    # -- plain Adam (+ optional L-BFGS) on one network ----------------------
    def _run_single(self, t0: float, t_ref: np.ndarray, ys_ref: np.ndarray) -> None:
        cfg, model, grid, history = self.cfg, self.model, self.grid, self.history
        adam, sched = adam_with_decay(model.parameters(), cfg)
        start = 0
        if cfg.checkpoint_every:
            cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
            saved = sorted(cfg.ckpt_path.glob("adam_*.pt"))
            if saved:
                state = torch.load(saved[-1], map_location=self.device)
                model.load_state_dict(state["model"]); adam.load_state_dict(state["adam"])
                sched.load_state_dict(state["sched"]); start = state["epoch"]
                history.resumed_from = start
        self.resumed = start

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
            self._snapshot(epoch, parts, final=last)

            lr = adam.param_groups[0]["lr"]
            before = flat_params(model) if logging else None
            adam.step()
            sched.step()
            history.loss.append(loss.item())

            if logging:
                history.record_step(epoch, model=model, loss=loss, residual=res, ic=ic,
                                    lr=lr, params_before=before)
            if epoch % cfg.eval_every == 0 or last:
                history.record_reference(epoch, float(np.mean((predict(model, t_ref) - ys_ref) ** 2)))

            if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
                torch.save({"model": model.state_dict(), "adam": adam.state_dict(),
                            "sched": sched.state_dict(), "epoch": epoch + 1},
                           cfg.ckpt_path / f"adam_{epoch + 1:06d}.pt")
            if cfg.print_every and (epoch % cfg.print_every == 0 or last):
                elapsed = time.perf_counter() - t0
                eta = elapsed / (epoch + 1) * (cfg.epochs - epoch - 1)
                print(f"[adam]  epoch {epoch + 1:>6}/{cfg.epochs} | loss {history.loss[-1]:.4e} | "
                      f"res {res.item():.4e} | ic {ic.item():.4e} | "
                      f"|grad| {history.grad_norm[-1]:.3e} | step {history.update_norm[-1]:.3e} | "
                      f"lr {lr:.2e} | ref_mse {history.ref_mse[-1]:.4e} | "
                      f"{elapsed:6.1f}s elapsed, ~{eta:5.0f}s left", flush=True)

        history.adam_iters = len(history.loss)
        history.wall_clock_s = time.perf_counter() - t0
        if cfg.print_every:
            print(f"[adam]  done in {time.perf_counter() - t0:.1f}s | "
                  f"final loss {history.loss[-1]:.4e} | best loss {min(history.loss):.4e}", flush=True)

        if cfg.lbfgs_iters > 0:
            def closure() -> Tensor:
                loss = pinn_loss(model, grid.clone().requires_grad_(True))
                history.loss.append(loss.item())
                n = len(history.loss) - history.adam_iters
                if cfg.print_every and n % 10 == 1:
                    print(f"[lbfgs] eval {n:>5} | loss {history.loss[-1]:.4e} | "
                          f"{time.perf_counter() - t0:6.1f}s elapsed", flush=True)
                return loss

            run_lbfgs(model.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
            history.wall_clock_s = time.perf_counter() - t0
            if cfg.print_every:
                print(f"[lbfgs] done after {len(history.loss) - history.adam_iters} evals | "
                      f"final loss {history.loss[-1]:.4e}", flush=True)

    # -- window by window, causal schedule ----------------------------------
    def _run_windows(self, t0: float, t_ref: np.ndarray, ys_ref: np.ndarray) -> None:
        """Window by window: Adam under each eps until min_i w_i > delta, then L-BFGS, then hand
        the end state to the next window. One window with an empty schedule is plain Adam."""
        cfg, model, grid, history = self.cfg, self.model, self.grid, self.history
        windowed = isinstance(model, WindowedPINN)     # False for a shared net: one sub, n_win rounds
        n_win = len(model.windows) if windowed else cfg.n_windows
        stages = list(cfg.causal_eps_schedule) or [None]
        cap = cfg.causal_max_iters if cfg.causal_eps_schedule else cfg.epochs
        device = grid.device

        for k in range(n_win):
            sub = model.windows[k] if windowed else model
            saved = cfg.ckpt_path / f"window_{k:02d}.pt"
            pts = grid[model.window_of(grid) == k] if hasattr(model, "window_of") else grid
            if saved.exists():
                sub.load_state_dict(torch.load(saved, map_location=device))
                if cfg.print_every:
                    print(f"[window] {k:>2} restored from {saved.name}", flush=True)
            else:
                if k and cfg.warm_start and windowed:
                    with torch.no_grad():
                        for p, q in zip(sub.net.parameters(), model.windows[k - 1].net.parameters()):
                            p.copy_(q)
                adam, sched = adam_with_decay(sub.parameters(), cfg)
                for eps in stages:
                    for _ in range(cap):
                        epoch = len(history.loss)
                        adam.zero_grad()
                        t = pts.clone().requires_grad_(True)
                        res, ic, parts = loss_terms(sub, t)
                        w = None
                        if eps is not None:
                            res, w = causal_loss(parts.r, t, eps)
                            history.min_w.append(float(w.min()))
                        loss = res + sub.gamma * ic if (cfg.ic == "soft" or sub.end_state is not None) else res
                        loss.backward()
                        self._snapshot(epoch, w=w)
                        logging = epoch % cfg.log_every == 0
                        before = flat_params(sub) if logging else None
                        lr = adam.param_groups[0]["lr"]
                        adam.step()
                        sched.step()
                        history.loss.append(loss.item())
                        if logging:
                            history.record_step(epoch, model=sub, loss=loss, residual=res, ic=ic,
                                                lr=lr, params_before=before)
                        if epoch % cfg.eval_every == 0:
                            history.record_reference(epoch, float(np.mean((predict(model, t_ref) - ys_ref) ** 2)))
                        if cfg.print_every and epoch % cfg.print_every == 0:
                            min_w = history.min_w[-1] if w is not None else 1.0
                            print(f"[adam]  window {k:>2} eps {eps} | epoch {epoch:>7} | loss {loss.item():.4e} "
                                  f"| min_w {min_w:.3f} | {time.perf_counter() - t0:7.1f}s", flush=True)
                        if w is not None and history.min_w[-1] > cfg.causal_delta:
                            break
                    if eps is not None:
                        history.eps_marks.append((len(history.loss), eps))
                history.adam_iters = len(history.loss)
                if cfg.lbfgs_iters > 0:
                    def closure() -> Tensor:
                        loss = pinn_loss(sub, pts.clone().requires_grad_(True))
                        history.loss.append(loss.item())
                        return loss

                    run_lbfgs(sub.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
                cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save(sub.state_dict(), saved)
                done = torch.ones(len(pts), dtype=grid.dtype) if stages[0] is not None else None
                self._snapshot(len(history.loss) - 1, w=done, final=True)
            history.window_marks.append(len(history.loss))
            if k + 1 < n_win:
                with torch.no_grad():
                    edge = torch.tensor([[model.edges[k + 1]]], dtype=grid.dtype, device=device)
                    model.set_window_start(k + 1, model.window_forward(k, edge)[0])
            if cfg.print_every and history.loss:
                print(f"[window] {k:>2} done | loss {history.loss[-1]:.4e} | "
                      f"{time.perf_counter() - t0:7.1f}s", flush=True)
        history.wall_clock_s = time.perf_counter() - t0
