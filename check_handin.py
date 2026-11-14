"""Hand-in self-check. Run inside the hand-in folder, in a fresh environment with only requirements.txt installed.

    python check_handin.py [path/to/test.csv] [expected_md5]

1. The organisers' check: sample 100,000 shots per basis (seed 1), decode, print LER and decode time.
2. If test.csv is given: decode it, write submission_check.csv and print its md5 (compare with the selected
   submission), then time decode() on it five times and report the median.
Uses only numpy and stim besides the decoder itself.
"""
import csv
import hashlib
import statistics
import sys
import time

import numpy as np
import stim

t0 = time.perf_counter()
from decoder import decode  # noqa: E402

print(f"import decoder: {time.perf_counter() - t0:.1f}s (not timed by the organisers)")

total = 0.0
for basis in "XZ":
    circuit = stim.Circuit.from_file(f"circuits/{basis}_circuit.stim")
    events, flips = circuit.compile_detector_sampler(seed=1).sample(100_000, separate_observables=True)
    start = time.perf_counter()
    predictions = np.asarray(decode(circuit, events))
    total += time.perf_counter() - start
    assert predictions.shape == flips.shape and np.isin(predictions, (0, 1)).all()
    print(f"{basis}: LER {np.mean(predictions != flips):.5f}")
print(f"decode time: {total:.2f} s")

if len(sys.argv) > 1:
    ids, basis_col, rows = [], [], []
    with open(sys.argv[1], newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            ids.append(r[0])
            basis_col.append(r[1])
            rows.append(r[2:72])
    det = np.array(rows, dtype=np.uint8).astype(bool)
    basis_col = np.array(basis_col)
    circ = {b: stim.Circuit.from_file(f"circuits/{b}_circuit.stim") for b in "XZ"}
    idx = {b: np.flatnonzero(basis_col == b) for b in "XZ"}
    pred = {b: np.asarray(decode(circ[b], det[idx[b]])) for b in "XZ"}
    with open("submission_check.csv", "w", newline="") as f:
        f.write("id,L0,L1,L2,L3\n")
        for b in "XZ":
            for i, p in zip(idx[b], pred[b]):
                f.write(f"{ids[i]},{p[0]},{p[1]},{p[2]},{p[3]}\n")
    md5 = hashlib.md5(open("submission_check.csv", "rb").read()).hexdigest()
    print(f"submission_check.csv md5 {md5}")
    if len(sys.argv) > 2:
        print("MATCHES the selected submission" if md5 == sys.argv[2] else f"DOES NOT MATCH expected {sys.argv[2]}")
    times = []
    for _ in range(5):
        start = time.perf_counter()
        again = {b: np.asarray(decode(circ[b], det[idx[b]])) for b in "XZ"}
        times.append(time.perf_counter() - start)
        assert all(np.array_equal(again[b], pred[b]) for b in "XZ"), "predictions changed between runs"
    print("test.csv decode times: " + ", ".join(f"{t:.3f}" for t in times) + f" | median {statistics.median(times):.3f} s")
