"""
Phase 1 baseline: Visual Cryptography (2,2) scheme
  1. Naor-Shamir basic VC      -> shares look like random noise
  2. Extended VC (meaningful)  -> shares look like cover images

Convention everywhere: 1 = BLACK, 0 = WHITE/transparent.
Stacking transparencies = bitwise OR.

Usage:
    python vc_baseline.py                          # runs with generated demo images
    python vc_baseline.py secret.png c1.png c2.png # runs with your own images
"""
import sys
import itertools
import numpy as np
import os
from PIL import Image, ImageDraw

rng = np.random.default_rng(42)

# ---------- 2x2 patterns ----------
# A pattern is a 2x2 array of 0/1. There are 16 possible patterns.
ALL_PATTERNS = [np.array(p).reshape(2, 2) for p in itertools.product([0, 1], repeat=4)]
TWO_BLACK = [p for p in ALL_PATTERNS if p.sum() == 2]  # 6 patterns


# ---------- preprocessing ----------
def load_binary(path_or_img, size):
    """Load image -> resize -> Floyd-Steinberg halftone -> 0/1 array (1 = black)."""
    img = Image.open(path_or_img) if isinstance(path_or_img, str) else path_or_img
    img = img.convert("L").resize(size)
    bw = img.convert("1")  # PIL uses Floyd-Steinberg dithering here
    return (np.array(bw) == 0).astype(np.uint8)  # black pixel -> 1


def to_image(arr):
    """0/1 array (1=black) -> PIL image."""
    return Image.fromarray(((1 - arr) * 255).astype(np.uint8))


def stack(s1, s2):
    """Stacking two transparencies = OR."""
    return np.bitwise_or(s1, s2)


# ---------- 1. Naor-Shamir basic VC ----------
def basic_vc(secret):
    """secret: 0/1 array (1=black). Returns share1, share2 (each 2x bigger)."""
    h, w = secret.shape
    s1 = np.zeros((2 * h, 2 * w), dtype=np.uint8)
    s2 = np.zeros_like(s1)
    for i in range(h):
        for j in range(w):
            p = TWO_BLACK[rng.integers(len(TWO_BLACK))]      # random pattern
            q = p if secret[i, j] == 0 else 1 - p            # same (white) / complement (black)
            s1[2*i:2*i+2, 2*j:2*j+2] = p
            s2[2*i:2*i+2, 2*j:2*j+2] = q
    return s1, s2


# ---------- 2. Extended VC (meaningful shares) ----------
# Cover pixel white -> share block has 2 black subpixels (50% dark)
# Cover pixel black -> share block has 3 black subpixels (75% dark)
# Secret white -> one pattern must contain the other   (stack darkness = max)
# Secret black -> patterns together cover all 4 cells  (stack darkness = 100%)
def _build_table():
    table = {}
    for s, c1, c2 in itertools.product([0, 1], repeat=3):
        n1, n2 = 2 + c1, 2 + c2
        valid = []
        for p in ALL_PATTERNS:
            if p.sum() != n1:
                continue
            for q in ALL_PATTERNS:
                if q.sum() != n2:
                    continue
                total = np.bitwise_or(p, q).sum()
                if s == 1 and total == 4:
                    valid.append((p, q))
                elif s == 0 and total == max(n1, n2):
                    valid.append((p, q))
        table[(s, c1, c2)] = valid
    return table

TABLE = _build_table()


def extended_vc(secret, cover1, cover2):
    """All three inputs: 0/1 arrays of the same shape (1=black). Shares are 2x bigger."""
    assert secret.shape == cover1.shape == cover2.shape, "resize inputs to the same size"
    h, w = secret.shape
    s1 = np.zeros((2 * h, 2 * w), dtype=np.uint8)
    s2 = np.zeros_like(s1)
    for i in range(h):
        for j in range(w):
            options = TABLE[(secret[i, j], cover1[i, j], cover2[i, j])]
            p, q = options[rng.integers(len(options))]
            s1[2*i:2*i+2, 2*j:2*j+2] = p
            s2[2*i:2*i+2, 2*j:2*j+2] = q
    return s1, s2


# ---------- metrics ----------
def contrast(stacked, secret):
    """alpha = (darkness of black secret areas - darkness of white secret areas)."""
    big = np.kron(secret, np.ones((2, 2), dtype=np.uint8))
    return stacked[big == 1].mean() - stacked[big == 0].mean()


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)  # inputs in 0..1


# ---------- demo images ----------
def make_demo(size=(150, 150)):
    secret = Image.new("L", size, 255)
    ImageDraw.Draw(secret).text((30, 55), "SECRET", fill=0)
    secret = secret.resize((size[0], size[1]))
    # bigger text: draw on small canvas then scale
    sm = Image.new("L", (40, 12), 255)
    ImageDraw.Draw(sm).text((2, 0), "SECRET", fill=0)
    secret = sm.resize(size, Image.NEAREST)

    x, y = np.meshgrid(np.linspace(0, 1, size[0]), np.linspace(0, 1, size[1]))
    c1 = Image.fromarray((255 * (0.15 + 0.7 * x)).astype(np.uint8))              # horizontal gradient
    c2 = Image.fromarray((255 * (0.15 + 0.7 * np.hypot(x - .5, y - .5) * 1.6).clip(0, 1)).astype(np.uint8))  # radial
    return secret, c1, c2


def main():
    size = (150, 150)
    if len(sys.argv) == 4:
        secret_img, c1_img, c2_img = sys.argv[1:4]
    else:
        secret_img, c1_img, c2_img = make_demo(size)

    secret = load_binary(secret_img, size)
    cover1 = load_binary(c1_img, size)
    cover2 = load_binary(c2_img, size)

    # Basic VC
    b1, b2 = basic_vc(secret)
    bs = stack(b1, b2)
    # Extended VC
    e1, e2 = extended_vc(secret, cover1, cover2)
    es = stack(e1, e2)
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "outputs")
    os.makedirs(OUT, exist_ok=True)
    for name, arr in [("secret", secret), ("basic_share1", b1), ("basic_share2", b2),
                      ("basic_stacked", bs), ("cover1_halftone", cover1), ("cover2_halftone", cover2),
                      ("meaningful_share1", e1), ("meaningful_share2", e2), ("meaningful_stacked", es)]:
        to_image(arr).save(os.path.join(OUT, f"{name}.png"))

    print(f"Secret size: {secret.shape}  ->  share size: {b1.shape}  (pixel expansion m = 4)")
    print(f"Basic VC     contrast alpha = {contrast(bs, secret):.2f}")
    print(f"Extended VC  contrast alpha = {contrast(es, secret):.2f}")
    # how close is each meaningful share to its (halftoned, 2x upscaled) cover?
    up1 = np.kron(cover1, np.ones((2, 2), dtype=np.uint8))
    print(f"Share1 darkness {e1.mean():.2f} vs cover1 darkness {up1.mean():.2f} "
          f"(shares are darker than cover by design: 50%/75% levels)")


if __name__ == "__main__":
    main()