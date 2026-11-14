# Superposed Devs Unobserved

## Hybrid decoder for the [[18,4,4]] qLDPC code: neural pre-decoder + exact maximum-likelihood decoder

`decode(circuit, detector_events)` in `decoder.py` combines two decoders, both written by us:

1. **Neural network** (`weights/X.npz`, `weights/Z.npz`, one per basis): a residual MLP (width 1024, 8 blocks,
   33.7M parameters) that maps the 70 detector bits to one logit per logical observable. It was trained with
   binary cross-entropy on shots sampled with Stim from the competition circuits.
2. **Exact maximum-likelihood decoder** (`exact.py`): every mechanism of the circuit's detector error model
   touches detectors in at most two adjacent time layers, so for each observable the signed sum over all error
   sets consistent with a shot, `P(no flip, shot) - P(flip, shot)`, is a chain of five table products swept
   through the six time layers (a transfer-matrix / forward algorithm). The tables are built from the detector
   error model with Walsh-Hadamard transforms. The sign of the sum is the Bayes-optimal prediction for that
   observable. There is no training and no approximation in this part.

The network decodes every shot first. Each prediction whose logit is closer to zero than the threshold in
`weights/hybrid.json` is recomputed by the exact decoder. On 800,000 freshly
sampled predictions this gives the same output as running the exact decoder on everything, at a fraction of
the cost.

## Files

- `decoder.py` – `decode()`; loads the weights, builds the exact decoder's tables and warms up the GPU at import.
- `exact.py` – the exact decoder.
- `weights/` – network weights and `hybrid.json` (threshold, chunk size, TF32 switches).
- `circuits/` – the two competition circuits, used at import to identify the basis and build the tables.
- `requirements.txt` – numpy, stim, torch, pinned.
- `check_handin.py` – self-check: LER on fresh shots, decode time, and the md5 of the submission it reproduces.
- `training/` – the code that created the weights.

Libraries: numpy, stim (loading circuits, sampling training shots, building the detector error model) and
PyTorch (tensor arithmetic and the network). No decoding package and no third-party decoder code is used.