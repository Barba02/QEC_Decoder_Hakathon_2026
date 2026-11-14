"""Exact maximum-likelihood decoding of the [[18,4,4]] memory circuits by a transfer-matrix sweep over time.

Idea
----
Every error mechanism of the circuit's detector error model flips detectors in a single time layer or in
two adjacent time layers (checked at build time). Split the mechanisms into blocks t = 0..T-2: block t holds
the mechanisms whose earliest detector lies in layer t (mechanisms living only in the last layer join the
last block). For one observable, define for block t the table

    K_t[u, v] = sum over subsets S of the block's mechanisms that flip exactly the detector pattern u in
                layer t and v in layer t+1, of  prob(S) * (-1)^(number of flips of the observable by S).

Then, for a shot with detector patterns D_0..D_{T-1},

    A = sum over all error sets consistent with the shot of prob * (-1)^(observable flip)
      = K_0[D_0, :] . P(D_1) K_1 . P(D_2) K_2 ... K_{T-2}[:, D_{T-1}]      (P(D) = XOR-shift by D)

A is proportional to P(no flip | shot) - P(flip | shot), so the observable is predicted flipped iff A < 0.
This is the exact Bayes-optimal decision for that observable; no training and no approximation is involved.

The tables are built with Walsh-Hadamard transforms: the transform of the signed distribution of a set of
independent mechanisms is the product of their individual transforms, which has a closed form.

Only numpy, stim (to read the circuit and its detector error model) and torch (tensor arithmetic) are used.
"""
import numpy as np
import stim
import torch


def merged_mechanisms(circuit: stim.Circuit):
    """{(frozenset(detectors), frozenset(observables)): probability}, mechanisms with equal effect merged."""
    cols = {}
    for inst in circuit.detector_error_model().flattened():
        if inst.type != "error":
            continue
        ds, ob = set(), set()
        for x in inst.targets_copy():
            if x.is_relative_detector_id():
                ds ^= {x.val}
            elif x.is_logical_observable_id():
                ob ^= {x.val}
        k, p = (frozenset(ds), frozenset(ob)), inst.args_copy()[0]
        q = cols.get(k, 0.0)
        cols[k] = q * (1 - p) + p * (1 - q)
    return cols


def _fwht_(x):
    """In-place unnormalised Walsh-Hadamard transform of a 1-D tensor of length 2^n."""
    n, h = x.numel(), 1
    while h < n:
        y = x.view(-1, 2, h)
        a, b = y[:, 0, :], y[:, 1, :]
        d = a - b
        a.add_(b)
        b.copy_(d)
        del d
        h *= 2
    return x


class ExactDecoder:
    def __init__(self, circuit: stim.Circuit, device, observables=None, tf32=False):
        self.device = torch.device(device)
        # tf32=True lets the GPU use TensorFloat-32 for the big table products: several times faster, inputs
        # rounded to ~3 significant digits. That can only change A's sign when the two outcomes are within
        # roughly 0.1% of each other in probability, i.e. where either answer is as good as the other.
        self.tf32 = bool(tf32) and self.device.type == "cuda"
        coords = circuit.get_detector_coordinates()
        T = int(max(v[2] for v in coords.values())) + 1
        self.layers = [sorted(d for d, v in coords.items() if int(v[2]) == t) for t in range(T)]
        pos = {d: (t, i) for t, ds in enumerate(self.layers) for i, d in enumerate(ds)}
        self.nb = T - 1
        self.ids = [torch.tensor(ids, device=self.device) for ids in self.layers]
        self.wts = [(1 << torch.arange(len(ids), device=self.device)) for ids in self.layers]
        self.ar = [torch.arange(1 << len(ids), device=self.device) for ids in self.layers]
        self.observables = list(range(circuit.num_observables)) if observables is None else list(observables)

        cols = merged_mechanisms(circuit)
        self.tables = {}
        cache = {}
        for obs in self.observables:
            terms = [dict() for _ in range(self.nb)]
            const = [0.0] * self.nb
            for (ds, ob), p in cols.items():
                ts = [pos[d][0] for d in ds]
                t = min(min(ts), self.nb - 1)
                if not ds or max(ts) - t > 1 or min(ts) < t:
                    raise ValueError("error mechanism does not fit the two-adjacent-layers structure")
                u = sum(1 << pos[d][1] for d in ds if pos[d][0] == t)
                v = sum(1 << pos[d][1] for d in ds if pos[d][0] == t + 1)
                w = float(np.log1p(-2.0 * p))
                terms[t][(u, v)] = terms[t].get((u, v), 0.0) + (-w if obs in ob else w)
                const[t] += w / 2
            row = []
            for t in range(self.nb):
                nu, nv = len(self.layers[t]), len(self.layers[t + 1])
                key = (nu, nv, round(const[t], 13), tuple(sorted((k, round(w, 13)) for k, w in terms[t].items())))
                if key not in cache:
                    cache[key] = self._table(terms[t], const[t], nu, nv)
                row.append(cache[key])
            self.tables[obs] = row

    def _table(self, term, const, nu, nv):
        keys = torch.tensor([(u << nv) | v for (u, v) in term], dtype=torch.long, device=self.device)
        vals = torch.tensor(list(term.values()), dtype=torch.float64, device=self.device)
        g = torch.zeros(1 << (nu + nv), dtype=torch.float64, device=self.device)
        g.index_put_((keys,), vals, accumulate=True)
        _fwht_(g)                          # sum_m s_m w_m (-1)^(a . x_m)
        g.mul_(-0.5).add_(const).exp_()    # product of the mechanisms' Fourier factors
        _fwht_(g)
        g.div_(g.numel())
        return g.to(torch.float32).view(1 << nu, 1 << nv).contiguous()

    @torch.inference_mode()
    def pack(self, det):
        """det: bool tensor (n, num_detectors) on the device -> list of per-layer integer patterns."""
        return [(det[:, ids].long() * w).sum(1) for ids, w in zip(self.ids, self.wts)]

    @torch.inference_mode()
    def signed_sums(self, D, obs, rows=None, chunk=8192):
        """A for the given observable. D: output of pack(); rows: optional index tensor of shots to decode."""
        K = self.tables[obs]
        if rows is not None:
            D = [d[rows] for d in D]
        n = D[0].numel()
        out = torch.empty(n, dtype=torch.float32, device=self.device)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            for i in range(0, n, chunk):
                sl = slice(i, i + chunk)
                f = K[0][D[0][sl]]
                f = f / f.abs().amax(1, keepdim=True).clamp_min(1e-30)
                for t in range(1, self.nb):
                    f = torch.gather(f, 1, self.ar[t][None, :] ^ D[t][sl][:, None]) @ K[t]
                    f = f / f.abs().amax(1, keepdim=True).clamp_min(1e-30)
                out[sl] = f.gather(1, D[self.nb][sl][:, None])[:, 0]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        return out
