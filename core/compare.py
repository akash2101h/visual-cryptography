"""
Results table for the report: hybrid scheme (several compression ratios) vs baselines.

Uses the same three images as pipeline.py:
    python compare.py                                   (paths from SECRET_PATH / COVER1_PATH / COVER2_PATH in pipeline.py)
    python compare.py secret.png cover1.png cover2.png  (or give them here)

Rows
  Mode A   : plain meaningful VC on the HALFTONED secret (eye-readable, no key)       [run]
  Bit-plane: send the 8-bit grayscale secret through VC with no compression (8 planes) [calculated]
  Hybrid   : compressive sensing + lattice + meaningful VC at ratio 0.25 / 0.5 / 0.75  [run]

Saved to ../data/outputs/results.csv
"""
import os
import sys
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

import pipeline as P
import lattice_lwe as lwe
from compressive_sensing import psnr
from visual_crypto import extended_vc, stack, load_binary, contrast
from bitpacking import decode_stacked


def share_resemblance(share, cover_gray):
    """How much a share looks like its cover: SSIM between the share (averaged per 2x2 block)
    and the contrast-enhanced cover. Higher = closer."""
    side = share.shape[0] // 2
    small = share.reshape(side, 2, side, 2).mean(axis=(1, 3))
    a = (small - small.min()) / (np.ptp(small) + 1e-9)
    return ssim(1 - a, np.array(cover_gray) / 255.0, data_range=1.0)


def mode_a(secret_gray, c1_src, c2_src):
    """Plain VC: halftone the secret (1 bit/pixel), hide it in two meaningful shares."""
    size = secret_gray.shape[0]
    secret_bits = load_binary(Image.fromarray((secret_gray * 255).astype(np.uint8)), (size, size))
    cov1 = load_binary(P._prep_cover(c1_src, (size, size)), (size, size))
    cov2 = load_binary(P._prep_cover(c2_src, (size, size)), (size, size))
    s1, s2 = extended_vc(secret_bits, cov1, cov2)
    stacked = stack(s1, s2)
    exact = np.array_equal(decode_stacked(stacked), secret_bits)
    eye_view = gaussian_filter(1.0 - secret_bits.astype(float), 1.0)     # what an eye roughly sees
    return dict(
        method="Mode A: plain VC (halftone secret)",
        share_area=s1.size / secret_gray.size,
        key="no", eye="yes (grey, alpha=%.2f)" % contrast(stacked, secret_bits),
        psnr=psnr(secret_gray, eye_view), ssim=ssim(secret_gray, eye_view, data_range=1.0),
        wrong_key="n/a", note=f"bits exact={exact}")


def hybrid(secret_p, c1_p, c2_p, public_key, secret_key, ratio):
    pkg = P.sender(secret_p, c1_p, c2_p, public_key, ratio=ratio)
    rec = P.receiver(pkg["share1"], pkg["share2"], pkg["ciphertext"], pkg["meta"], secret_key)
    _, other_key = lwe.keygen()
    bad = P.receiver(pkg["share1"], pkg["share2"], pkg["ciphertext"], pkg["meta"], other_key)
    side = pkg["share1"].shape[0] // 2
    res1 = share_resemblance(pkg["share1"], P._prep_cover(c1_p, (side, side)))
    res2 = share_resemblance(pkg["share2"], P._prep_cover(c2_p, (side, side)))
    sec = pkg["secret"]
    return dict(
        method=f"Hybrid CS+LWE+VC (ratio {ratio})",
        share_area=pkg["share1"].size / sec.size,
        key="yes (lattice)", eye="no (computer decode)",
        psnr=psnr(sec, rec), ssim=ssim(sec, rec, data_range=1.0),
        wrong_key=f"{psnr(sec, bad):.1f} / {ssim(sec, bad, data_range=1.0):.2f}",
        note=f"share-vs-cover SSIM {res1:.2f}/{res2:.2f}")


def main():
    a = sys.argv[1:]
    secret_p = r"C:\Users\agk21\OneDrive\Pictures\Screenshots\Screenshot 2026-10-12 234536.png"
    c1_p = r"C:\Users\agk21\OneDrive\Pictures\Screenshots\Screenshot 2026-10-13 001628.png" 
    c2_p = r"C:\Users\agk21\OneDrive\Pictures\Screenshots\Screenshot 2026-10-13 001752.png"
    if not (secret_p and c1_p and c2_p):
        sys.exit("Give three image paths (edit them in pipeline.py or pass them on the command line).")

    public_key, secret_key = P.get_keys()
    secret_gray = np.array(P._gray(secret_p, (P.SIZE, P.SIZE))) / 255.0

    rows = [mode_a(secret_gray, c1_p, c2_p)]
    rows.append(dict(method="Bit-plane VC, no compression (calculated)", share_area=8 * 4.0,
                     key="no", eye="no (8 bit-planes)", psnr=float("inf"), ssim=1.0,
                     wrong_key="n/a", note="lossless; 8 share pairs; not run"))
    for r in (0.25, 0.5, 0.75):
        rows.append(hybrid(secret_p, c1_p, c2_p, public_key, secret_key, r))

    print(f"\nSecret {P.SIZE}x{P.SIZE}\n")
    print(f"{'Method':<42}{'Share area':>11}{'Key':>15}{'PSNR':>9}{'SSIM':>7}{'Wrong key PSNR/SSIM':>21}")
    print("-" * 105)
    for r in rows:
        ps = "inf" if r["psnr"] == float("inf") else f"{r['psnr']:.1f}"
        print(f"{r['method']:<42}{r['share_area']:>10.0f}x{r['key']:>15}{ps:>9}{r['ssim']:>7.2f}{r['wrong_key']:>21}")
    print("\nWrong key = image rebuilt by someone holding both shares but a different lattice key.")
    print("Judge it mainly by SSIM (near 0 = no structure left); PSNR depends on how bright the secret is.")
    print("\nNotes:")
    for r in rows:
        print(f"  {r['method']}: eye-readable = {r['eye']}; {r['note']}")

    path = os.path.join(P.OUT_DIR, "results.csv")
    os.makedirs(P.OUT_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "share_area_x", "key_needed", "eye_readable", "psnr_db", "ssim", "wrong_key_psnr_dB_slash_ssim", "note"])
        for r in rows:
            w.writerow([r["method"], f"{r['share_area']:.1f}", r["key"], r["eye"],
                        "inf" if r["psnr"] == float("inf") else f"{r['psnr']:.2f}",
                        f"{r['ssim']:.3f}", r["wrong_key"], r["note"]])
    print(f"\nSaved: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()