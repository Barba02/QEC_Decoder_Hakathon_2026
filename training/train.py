"""Train a neural decoder for one basis of the [[18,4,4]] competition circuit.

Data is sampled on the fly with Stim from the competition circuit (never reused), so there is
no overfitting to a fixed training set. The model maps 70 detector bits -> 4 logits and is trained
with BCE, whose optimum is P(L_i = 1 | detectors): exactly what the per-bit LER metric rewards.

Weights are saved as .npz so decoder.py can run inference with numpy only.

Example (GPU):
    python train.py --basis X --device cuda --steps 200000 --batch 16384 --width 1024 --depth 8 --workers 16
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import stim
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUIT_DIR = os.path.join(HERE, "..", "data", "test")


# ----------------------------------------------------------------------------- data
def _worker(circuit_path, seed, chunk, queue):
    circuit = stim.Circuit.from_file(circuit_path)
    sampler = circuit.compile_detector_sampler(seed=seed)
    while True:
        det, obs = sampler.sample(chunk, separate_observables=True, bit_packed=True)
        queue.put((det, obs))


class Stream:
    """Background Stim sampling in several processes; yields (det uint8[n,70], obs uint8[n,4]).

    Each worker has its own seed and its own queue, and the queues are read round-robin, so the
    sequence of training batches depends only on (--seed, --workers, --batch), not on process timing.
    """

    def __init__(self, circuit_path, workers, chunk, seed):
        ctx = mp.get_context("spawn")
        self.queues = [ctx.Queue(maxsize=4) for _ in range(workers)]
        self.procs = [ctx.Process(target=_worker, args=(circuit_path, seed + 1000 + i, chunk, q), daemon=True)
                      for i, q in enumerate(self.queues)]
        for p in self.procs:
            p.start()
        self.turn = 0

    def get(self):
        det, obs = self.queues[self.turn].get()
        self.turn = (self.turn + 1) % len(self.queues)
        det = np.unpackbits(det, axis=1, bitorder="little", count=70)
        obs = np.unpackbits(obs, axis=1, bitorder="little", count=4)
        return det, obs

    def close(self):
        for p in self.procs:
            p.terminate()


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, width, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * expansion)
        self.fc2 = nn.Linear(width * expansion, width)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


def cumulative_matrix(circuit):
    """M[i, j] = 1 if detectors i and j belong to the same check and i is not later than j.

    (det @ M) % 2 is then, for every detector, the XOR of all detectors of its check up to that round:
    the accumulated syndrome of the check, i.e. which checks are currently violated rather than which changed.
    """
    coords = circuit.get_detector_coordinates()
    n = circuit.num_detectors
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if coords[i][:2] == coords[j][:2] and coords[i][2] <= coords[j][2]:
                m[i, j] = 1
    return m


class ResMLP(nn.Module):
    def __init__(self, width, depth, n_in=70, n_out=4, feat_m=None):
        super().__init__()
        if feat_m is not None:  # extra inputs: accumulated syndromes
            self.register_buffer("feat_M", torch.from_numpy(feat_m))
            n_in = n_in + feat_m.shape[1]
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, n_out)

    def forward(self, x):  # x: 0/1 floats
        if hasattr(self, "feat_M"):
            x = torch.cat([x, torch.remainder(x @ self.feat_M, 2)], dim=1)
        h = self.inp(2 * x - 1)
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h))


def export_npz(model, path, meta):
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path, meta=json.dumps(meta), **sd)


# ----------------------------------------------------------------------------- train
@torch.no_grad()
def evaluate(model, det, obs, device, bs=65536):
    model.eval()
    wrong = 0
    for i in range(0, len(det), bs):
        x = torch.from_numpy(det[i:i + bs]).to(device).float()
        pred = (model(x) > 0).cpu().numpy()
        wrong += (pred != obs[i:i + bs]).sum()
    model.train()
    return wrong / obs.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True, choices=["X", "Z"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--features", default="none", choices=["none", "cum"],
                    help="cum: also feed the accumulated syndrome of every check")
    ap.add_argument("--init", default=None, help="start from the weights in this .npz (same width/depth)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = args.out or os.path.join(HERE, "weights", f"{args.basis}_w{args.width}_d{args.depth}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    circuit_path = os.path.abspath(os.path.join(CIRCUIT_DIR, f"{args.basis}_circuit.stim"))

    # fixed validation set (seed distinct from training workers)
    val_det, val_obs = stim.Circuit.from_file(circuit_path).compile_detector_sampler(seed=123456 + args.seed) \
        .sample(args.val, separate_observables=True)
    print(f"val all-zeros LER {val_obs.mean():.5f}", flush=True)

    device = torch.device(args.device)
    feat_m = cumulative_matrix(stim.Circuit.from_file(circuit_path)) if args.features == "cum" else None
    model = ResMLP(args.width, args.depth, feat_m=feat_m).to(device)
    if args.init:
        z = np.load(args.init)
        model.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta"})
        print(f"initialised from {args.init}: val LER {evaluate(model, val_det, val_obs, device):.5f}", flush=True)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    fwd = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, min(1000, args.steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps))))
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"

    stream = Stream(circuit_path, args.workers, args.batch, args.seed * 100)
    buf_det, buf_obs = np.zeros((0, 70), np.uint8), np.zeros((0, 4), np.uint8)
    best, t0, run_loss = 1.0, time.time(), 0.0
    meta = dict(vars(args), arch="ResMLP", gelu="tanh")
    try:
        for step in range(1, args.steps + 1):
            while len(buf_det) < args.batch:
                d, o = stream.get()
                buf_det, buf_obs = np.concatenate([buf_det, d]), np.concatenate([buf_obs, o])
            x = torch.from_numpy(buf_det[:args.batch]).to(device, non_blocking=True).float()
            y = torch.from_numpy(buf_obs[:args.batch]).to(device, non_blocking=True).float()
            buf_det, buf_obs = buf_det[args.batch:], buf_obs[args.batch:]

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(fwd(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()

            if step % args.eval_every == 0 or step == args.steps:
                ler = evaluate(model, val_det, val_obs, device)
                export_npz(model, out.replace(".npz", "_last.npz"), dict(meta, step=step, val_ler=float(ler)))
                shots = step * args.batch
                flag = ""
                if ler < best:
                    best = ler
                    export_npz(model, out, dict(meta, step=step, val_ler=float(ler)))
                    flag = " *saved"
                print(f"step {step:7d} | shots {shots / 1e6:8.1f}M | loss {run_loss / args.eval_every:.5f} | "
                      f"val LER {ler:.5f} | lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:7.0f}s{flag}",
                      flush=True)
                run_loss = 0.0
    finally:
        stream.close()
    print(f"best val LER {best:.5f} -> {out}")


if __name__ == "__main__":
    main()
