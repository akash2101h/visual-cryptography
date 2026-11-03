"""
Compressive Sensing module (block-based).

IDEA
  Split the image into small blocks (default 16x16). For each block x (flattened, n = 256 numbers):
        y = Phi @ x          Phi is an m x n random matrix, m = ratio * n  (m < n  -> fewer numbers)
  Phi is generated from a secret SEED, so the receiver can rebuild it. The seed is what the
  lattice (LWE) module will protect later.

  Reconstruction uses the fact that natural images are SPARSE in the DCT domain (most DCT
  coefficients are ~0). We solve   min ||s||_1   s.t.  Phi * IDCT(s) ~= y   with FISTA.

Pixel values are handled in range 0..1.
"""
import numpy as np
from scipy.fft import dct


# ---------- helpers ----------
def _dct2_matrix(block):
    """Orthonormal 2D-DCT as one (n x n) matrix, n = block*block. s = D2 @ x, x = D2.T @ s."""
    D = dct(np.eye(block), norm="ortho", axis=0)  # 1D orthonormal DCT matrix
    return np.kron(D, D)


def make_phi(seed: int, m: int, n: int):
    """Random Gaussian measurement matrix, fully determined by the seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n)) / np.sqrt(m)


def block_permutation(seed, n_blocks):
    """Seed-dependent shuffle of the block order.
    Without it, the size of each block's measurements reveals how bright that block is (energy leaks),
    so even a wrong seed would give a coarse silhouette of the image. Transmitted row i holds block perm[i]."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    return rng.permutation(n_blocks)


def _to_blocks(img, b):
    h, w = img.shape
    assert h % b == 0 and w % b == 0, "image size must be a multiple of the block size"
    return (img.reshape(h // b, b, w // b, b).swapaxes(1, 2).reshape(-1, b * b))


def _from_blocks(blocks, shape, b):
    h, w = shape
    return (blocks.reshape(h // b, w // b, b, b).swapaxes(1, 2).reshape(h, w))


# ---------- quantization (measurements are real numbers; VC needs bits) ----------
def quantize(y, bits=8):
    """Map real measurements to integers 0..2^bits-1. Returns (q, lo, hi)."""
    lo, hi = float(y.min()), float(y.max())
    levels = 2 ** bits - 1
    q = np.round((y - lo) / (hi - lo) * levels).astype(np.uint16)
    return q, lo, hi


def dequantize(q, lo, hi, bits=8):
    levels = 2 ** bits - 1
    return q.astype(np.float64) / levels * (hi - lo) + lo


# ---------- main API ----------
def compress(img, ratio=0.5, seed=12345, block=16, bits=8):
    """
    img   : 2D float array in 0..1
    returns (q, meta)
        q    : integer array (n_blocks x m) - the quantized measurements (this is the payload)
        meta : dict with what the receiver needs besides the seed
    """
    n = block * block
    m = int(round(ratio * n))
    X = _to_blocks(img.astype(np.float64), block)          # (nb, n)
    Phi = make_phi(seed, m, n)
    Y = X @ Phi.T                                           # (nb, m)
    Y = Y[block_permutation(seed, Y.shape[0])]              # hide which measurements belong to which block
    q, lo, hi = quantize(Y, bits)
    meta = dict(shape=img.shape, block=block, m=m, bits=bits, lo=lo, hi=hi)
    return q, meta


def reconstruct(q, meta, seed=12345, iters=300, lam=0.005):
    """Rebuild the image from quantized measurements + seed (FISTA, L1 in DCT domain)."""
    b, m = meta["block"], meta["m"]
    n = b * b
    Y = dequantize(q, meta["lo"], meta["hi"], meta["bits"])  # (nb, m), rows in shuffled order
    perm = block_permutation(seed, Y.shape[0])
    Y_ordered = np.empty_like(Y)
    Y_ordered[perm] = Y                                      # undo the shuffle (wrong seed -> wrong order)
    Y = Y_ordered
    Phi = make_phi(seed, m, n)
    D2 = _dct2_matrix(b)
    A = Phi @ D2.T                                           # y = A @ s
    L = np.linalg.norm(A, 2) ** 2                            # Lipschitz constant

    S = Y @ A / L                                            # starting point (A^T y / L)
    Z, t = S.copy(), 1.0
    for _ in range(iters):
        grad = (Z @ A.T - Y) @ A
        S_new = Z - grad / L
        # soft-threshold all coefficients except the DC term (index 0)
        thr = lam / L
        dc = S_new[:, 0].copy()
        S_new = np.sign(S_new) * np.maximum(np.abs(S_new) - thr, 0)
        S_new[:, 0] = dc
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Z = S_new + ((t - 1) / t_new) * (S_new - S)
        S, t = S_new, t_new

    X = S @ D2                                               # x = D2.T @ s  (row form: s @ D2)
    return np.clip(_from_blocks(X, meta["shape"], b), 0, 1)


# ---------- self test ----------
def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


if __name__ == "__main__":
    from skimage import data
    from skimage.transform import resize

    img = resize(data.camera() / 255.0, (128, 128), anti_aliasing=True)
    print("Cameraman 128x128, block 16, 8-bit measurements\n")
    print(f"{'ratio':>6} {'bits':>5} {'PSNR dB':>8} {'payload bits':>13} {'vs raw 8bpp':>12}")
    for bits in (8, 6, 4):
        for ratio in (0.25, 0.5, 0.75):
            q, meta = compress(img, ratio=ratio, bits=bits)
            rec = reconstruct(q, meta)
            payload = q.size * bits
            print(f"{ratio:>6} {bits:>5} {psnr(img, rec):>8.2f} {payload:>13} {payload / (img.size * 8):>11.0%}")