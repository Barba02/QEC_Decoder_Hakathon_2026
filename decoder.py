"""Neural decoder for the [[18,4,4]] competition circuits.

A residual MLP maps the 70 detector bits to 4 logits, one per logical observable; a logit > 0 means
"this observable flipped". The weights in weights/ were trained by train.py on Stim samples of the
competition circuits (one model per basis).

Inference runs with PyTorch (GPU if available, otherwise CPU) in float32. Everything expensive
(loading weights, moving them to the device, building the exact decoder's tables, CUDA warm-up) happens
at import. If PyTorch is missing, a slower numpy implementation of the network alone is used.

Hybrid mode (weights/hybrid.json): predictions the network is not certain about are recomputed by the exact
maximum-likelihood decoder in exact.py, which is optimal but more expensive per shot.
"""
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede torch import

import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))

def _load(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return meta, {k: z[k].astype(np.float32) for k in z.files if k != "meta"}

# Which weight files to use per basis. Default: weights/X.npz and weights/Z.npz. If weights/ensemble.json
# exists, e.g. {"X": ["X.npz", "X_b.npz"], "Z": ["Z.npz", "Z_b.npz"]}, the logits of the listed models are averaged.
_ENS_PATH = os.path.join(HERE, "weights", "ensemble.json")
_FILES = json.load(open(_ENS_PATH)) if os.path.exists(_ENS_PATH) else {b: [f"{b}.npz"] for b in "XZ"}
_MODELS = {b: [_load(os.path.join(HERE, "weights", f)) for f in _FILES[b]] for b in "XZ"}
_CIRCUITS = {b: str(stim.Circuit.from_file(os.path.join(HERE, "circuits", f"{b}_circuit.stim"))) for b in "XZ"}

def _basis(circuit: stim.Circuit) -> str:
    text = str(circuit)
    for b, ref in _CIRCUITS.items():
        if text == ref:
            return b
    # Fallback if the text differs: the first-round detectors sit on the X checks (odd x coordinate)
    # in the X circuit and on the Z checks (even x coordinate) in the Z circuit.
    return "X" if int(circuit.get_detector_coordinates([0])[0][0]) % 2 == 1 else "Z"

# ----------------------------------------------------------------------------- torch path (default)
_TORCH = None
if os.environ.get("DECODER_FORCE_NUMPY") != "1":
    try:
        import torch
        import torch.nn.functional as F

        _DEV = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("DECODER_NO_GPU") != "1" else "cpu")
        if _DEV.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
        _TORCH = {b: [(meta, {k: torch.from_numpy(v).to(_DEV) for k, v in w.items()}) for meta, w in ms]
                  for b, ms in _MODELS.items()}
    except ImportError:
        _TORCH = None

def _forward_torch(w, meta, x):
    act = "tanh" if meta.get("gelu") == "tanh" else "none"
    if "feat_M" in w:  # model trained with accumulated-syndrome inputs
        x = torch.cat([x, torch.remainder(x @ w["feat_M"], 2)], dim=1)
    h = (2 * x - 1) @ w["inp.weight"].T + w["inp.bias"]
    for i in range(meta["depth"]):
        p = f"blocks.{i}."
        u = F.layer_norm(h, h.shape[-1:], w[p + "norm.weight"], w[p + "norm.bias"], 1e-5)
        u = F.gelu(u @ w[p + "fc1.weight"].T + w[p + "fc1.bias"], approximate=act)
        h = h + u @ w[p + "fc2.weight"].T + w[p + "fc2.bias"]
    h = F.layer_norm(h, h.shape[-1:], w["norm.weight"], w["norm.bias"], 1e-5)
    return h @ w["out.weight"].T + w["out.bias"]

def _logits_device(basis, x, batch=50000):
    """x: bool tensor (n, 70) on the device -> float32 logits (n, 4) on the device."""
    out = None
    for meta, w in _TORCH[basis]:  # fixed order, so the sum is reproducible
        o = torch.cat([_forward_torch(w, meta, x[i:i + batch].float()) for i in range(0, len(x), batch)])
        out = o if out is None else out + o
    return out / len(_TORCH[basis])

def _to_device(detector_events):
    return torch.from_numpy(np.ascontiguousarray(detector_events, dtype=np.bool_)).to(_DEV)

def _logits_torch(basis, detector_events):
    with torch.inference_mode():
        return _logits_device(basis, _to_device(detector_events)).cpu().numpy()

# ----------------------------------------------------------------------------- exact decoder (hybrid mode)
# If weights/hybrid.json exists, e.g. {"threshold": 6.0}, every prediction whose network logit is closer to
# zero than the threshold is recomputed by the exact maximum-likelihood decoder in exact.py. The network is
# then only a fast filter for the predictions it is certain about; a huge threshold means "exact everywhere".
# The exact decoder's tables are built here, at import.
_HYB_PATH = os.path.join(HERE, "weights", "hybrid.json")
_HYBRID = json.load(open(_HYB_PATH)) if os.path.exists(_HYB_PATH) else None
_EXACT = None
if _HYBRID is not None and _TORCH is not None:
    import sys

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from exact import ExactDecoder

    _EXACT_OBS = [int(o) for o in os.environ.get("DECODER_EXACT_OBS", "0,1,2,3").split(",")]  # debugging aid
    _EXACT = {b: ExactDecoder(stim.Circuit(_CIRCUITS[b]), _DEV, observables=_EXACT_OBS,
                              tf32=bool(_HYBRID.get("tf32", False))) for b in "XZ"}

def _decode_hybrid(basis, detector_events):
    ex, th, chunk = _EXACT[basis], float(_HYBRID["threshold"]), int(_HYBRID.get("chunk", 8192))
    with torch.inference_mode():
        x = _to_device(detector_events)
        # "nn_tf32": TensorFloat-32 products for the network too. It only filters, and at the threshold the
        # network and the exact decoder agree, so tiny changes in its logits do not change the output.
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = bool(_HYBRID.get("nn_tf32", False)) and _DEV.type == "cuda"
        try:
            lg = _logits_device(basis, x)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        pred = lg > 0
        D = ex.pack(x)
        for j in ex.observables:
            rows = torch.nonzero(lg[:, j].abs() < th)[:, 0]
            if rows.numel():
                A = ex.signed_sums(D, j, rows=rows, chunk=chunk)
                pred[rows, j] = torch.where(A != 0, A < 0, pred[rows, j])  # A == 0: keep the network's answer
        return pred.cpu().numpy().astype(np.uint8)

if _TORCH is not None:  # warm-up at import: CUDA init and kernel selection happen before any timing
    for _b in _TORCH:
        _logits_torch(_b, np.zeros((2048, 70), dtype=bool))
        if _EXACT is not None:
            _n = 16384 if _DEV.type == "cuda" else 32
            _decode_hybrid(_b, np.zeros((_n, 70), dtype=bool))
            _rng = np.random.default_rng(0)   # random detector patterns: many go through the exact decoder
            _decode_hybrid(_b, _rng.random((_n, 70)) < 0.1)

# ----------------------------------------------------------------------------- numpy path (fallback)
def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x-mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def _gelu(x, approximate):
    if approximate == "tanh":
        return 0.5 * x * (1.0 + np.tanh(np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x ** 3)))
    from scipy.special import erf  # only for models trained with exact GELU
    return 0.5 * x * (1.0 + erf(x / np.float32(1.4142135623730951)))

def _forward(w, meta, x):
    act = meta.get("gelu", "none")
    if "feat_M" in w:
        x = np.concatenate([x, np.remainder(x @ w["feat_M"], 2)], axis=1)
    h = (2 * x - 1) @ w["inp.weight"].T + w["inp.bias"]
    for i in range(meta["depth"]):
        p = f"blocks.{i}."
        u = _layernorm(h, w[p + "norm.weight"], w[p + "norm.bias"])
        u = _gelu(u @ w[p + "fc1.weight"].T + w[p + "fc1.bias"], act)
        h = h + u @ w[p + "fc2.weight"].T + w[p + "fc2.bias"]
    h = _layernorm(h, w["norm.weight"], w["norm.bias"])
    return h @ w["out.weight"].T + w["out.bias"]

# ----------------------------------------------------------------------------- public API
def logits(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    basis = _basis(circuit)
    if _TORCH is not None:
        return _logits_torch(basis, detector_events)
    x = np.asarray(detector_events, dtype=np.float32)
    outs = [np.concatenate([_forward(w, meta, x[i:i + 20000]) for i in range(0, len(x), 20000)])
            for meta, w in _MODELS[basis]]
    return sum(outs[1:], outs[0]) / np.float32(len(outs))

def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """detector_events: bool (shots, 70) -> uint8 (shots, 4)."""
    if _EXACT is not None:
        return _decode_hybrid(_basis(circuit), detector_events)
    return (logits(circuit, detector_events) > 0).astype(np.uint8)
