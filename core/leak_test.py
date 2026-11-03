"""
Wrong-seed leakage test for the compressive-sensing step.

Question: if someone rebuilds the image with the WRONG seed, does the result still show the coarse
layout (which 16x16 blocks are bright or dark) of the secret?

Measure: correlation between the brightness of each 16x16 block of the secret and of the wrong-seed
reconstruction. Near 0 = no layout leaks; near 1 = the silhouette leaks.
Compared: (a) without the seed-keyed block shuffle  (b) with it (what the project uses).

Run:  python leak_test.py secret.png [trials]
"""
import sys
import numpy as np
from PIL import Image

import compressive_sensing as cs
import lattice_lwe as lwe


def block_means(img, b=16):
    h, w = img.shape
    return img.reshape(h // b, b, w // b, b).mean(axis=(1, 3)).ravel()


def run(secret, trials):
    vals = []
    for _ in range(trials):
        seed, other = lwe.random_seed(), lwe.random_seed()
        q, meta = cs.compress(secret, ratio=0.5, seed=seed, bits=8)
        wrong = cs.reconstruct(q, meta, seed=other, iters=100)
        vals.append(np.corrcoef(block_means(secret), block_means(wrong))[0, 1])
    return np.mean(vals), np.std(vals)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python leak_test.py secret.png [trials]")
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    img = Image.open(sys.argv[1]).convert("L").resize((128, 128), Image.LANCZOS)
    secret = np.array(img) / 255.0

    real_perm = cs.block_permutation
    cs.block_permutation = lambda seed, n: np.arange(n)          # switch the shuffle OFF
    m0, s0 = run(secret, trials)
    cs.block_permutation = real_perm                              # switch it back ON
    m1, s1 = run(secret, trials)

    print(f"Wrong-seed block-brightness correlation, {trials} trials each")
    print(f"  without block shuffle : {m0:+.2f} (std {s0:.2f})")
    print(f"  with block shuffle    : {m1:+.2f} (std {s1:.2f})   <- used in this project")


   