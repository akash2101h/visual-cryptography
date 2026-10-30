"""
Web demo: Visually Meaningful Cryptography with Compressive Sensing + Lattice (LWE)

Run locally :  streamlit run app.py
Folder layout expected (app.py sits next to the core folder):
    app.py
    requirements.txt
    core/  (visual_crypto.py, compressive_sensing.py, bitpacking.py, lattice_lwe.py, pipeline.py)
"""
import os
import sys
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
import pipeline as P                       # noqa: E402
import lattice_lwe as lwe                  # noqa: E402
from compressive_sensing import psnr       # noqa: E402
from visual_crypto import to_image, stack  # noqa: E402


def gray_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def run_hybrid(secret_img, c1_img, c2_img, ratio, keys):
    """Runs sender and receiver in memory. Returns everything the page needs to display."""
    public_key, secret_key = keys
    pkg = P.sender(secret_img, c1_img, c2_img, public_key, ratio=ratio)
    args = (pkg["share1"], pkg["share2"], pkg["ciphertext"], pkg["meta"])
    recovered = P.receiver(*args, secret_key)
    _, other_key = lwe.keygen()
    wrong = P.receiver(*args, other_key)
    sec = pkg["secret"]
    return dict(
        secret=sec, recovered=recovered, wrong=wrong,
        share1=to_image(pkg["share1"]), share2=to_image(pkg["share2"]),
        stacked=to_image(stack(pkg["share1"], pkg["share2"])),
        psnr=psnr(sec, recovered), ssim=ssim(sec, recovered, data_range=1.0),
        wrong_psnr=psnr(sec, wrong),
        share_area=pkg["share1"].size / sec.size,
        ct_kb=lwe.ciphertext_size_bytes(pkg["ciphertext"]) / 1024,
    )


def main():
    st.set_page_config(page_title="Visually Meaningful Cryptography", layout="wide")
    st.title("Visually Meaningful Cryptography")
    st.caption("Compressive Sensing + Lattice (LWE) encryption + meaningful visual cryptography")
    st.info("Educational demo. Uploaded images are processed in memory, but this is a public web "
            "server: do not upload sensitive images.")

    if "keys" not in st.session_state:
        st.session_state.keys = lwe.keygen()      # receiver's lattice key pair, created per visit

    with st.sidebar:
        st.header("Inputs")
        secret_f = st.file_uploader("1. Secret image (to hide)", type=["png", "jpg", "jpeg", "bmp"])
        c1_f = st.file_uploader("2. Cover image 1 (Share 1 looks like this)", type=["png", "jpg", "jpeg", "bmp"])
        c2_f = st.file_uploader("3. Cover image 2 (Share 2 looks like this)", type=["png", "jpg", "jpeg", "bmp"])
        ratio = st.select_slider("Compression ratio", options=[0.25, 0.5, 0.75], value=0.5)
        run = st.button("Encrypt and recover", type="primary")

    if not run:
        st.write("Upload three images in the sidebar and press **Encrypt and recover**.")
        st.markdown("""
**How it works**
1. The secret is compressed with *compressive sensing*, using a random 128-bit seed.
2. The measurements are written as a black-and-white image and split into **two shares** that look like your cover images.
3. The seed is encrypted with *lattice (LWE)* cryptography. Both shares **and** the lattice key are needed to recover the secret.
""")
        return
    if not (secret_f and c1_f and c2_f):
        st.warning("Please upload all three images.")
        return

    with st.spinner("Running the pipeline..."):
        r = run_hybrid(Image.open(secret_f), Image.open(c1_f), Image.open(c2_f), ratio, st.session_state.keys)

    st.subheader("Results")
    m = st.columns(5)
    m[0].metric("PSNR (correct key)", f"{r['psnr']:.1f} dB")
    m[1].metric("SSIM", f"{r['ssim']:.2f}")
    m[2].metric("PSNR (wrong key)", f"{r['wrong_psnr']:.1f} dB")
    m[3].metric("Share size", f"{r['share_area']:.0f}x secret")
    m[4].metric("Lattice ciphertext", f"{r['ct_kb']:.0f} KB")

    st.subheader("Sender: the two shares")
    a, b, c = st.columns(3)
    a.image(r["share1"], caption="Share 1 (looks like cover 1)", use_container_width=True)
    b.image(r["share2"], caption="Share 2 (looks like cover 2)", use_container_width=True)
    c.image(r["stacked"], caption="Shares stacked: a bit pattern, decoded by computer", use_container_width=True)

    st.subheader("Receiver: recovering the secret")
    d, e, f = st.columns(3)
    d.image(gray_pil(r["secret"]), caption="Original secret", use_container_width=True)
    e.image(gray_pil(r["recovered"]), caption="Recovered with the correct lattice key", use_container_width=True)
    f.image(gray_pil(r["wrong"]), caption="Attacker with both shares, wrong lattice key", use_container_width=True)
    st.caption("Tip: open the shares at full size to see the cover image; the dot pattern blends at normal viewing size.")


if __name__ == "__main__":
    main()