"""
Full hybrid pipeline:  Compressive Sensing + Lattice (LWE) + Visually Meaningful VC

SENDER   (secret image, cover1, cover2, receiver's PUBLIC key)
   1. random 128-bit seed                       -> lattice_lwe.random_seed()
   2. compressive sensing with that seed        -> compressive_sensing.compress()
   3. measurements -> bits -> binary image      -> bitpacking.to_binary_image()
   4. meaningful VC: two shares like the covers -> visual_crypto.extended_vc()
   5. encrypt the seed with the public key      -> lattice_lwe.encrypt_seed()
   PACKAGE sent to receiver: share1.png, share2.png, ciphertext.npz, meta.json
   (the seed itself is NOT in the package)

RECEIVER (share1, share2, ciphertext, meta, PRIVATE key)
   1. stack shares, decode bits                 -> bitpacking.decode_stacked()
   2. bits -> measurements                      -> bitpacking.from_binary_image()
   3. decrypt the seed with the private key     -> lattice_lwe.decrypt_seed()
   4. rebuild the secret image                  -> compressive_sensing.reconstruct()

HOW TO GIVE YOUR IMAGES (either way):
   A) edit the three paths below (keep the r before the quotes), then run:  python pipeline.py
   B) pass them on the command line:  python pipeline.py secret.png cover1.png cover2.png [ratio]
"""
import os
import sys
import json
import time
import numpy as np
from PIL import Image, ImageOps

import lattice_lwe as lwe
from compressive_sensing import compress, reconstruct, psnr
from bitpacking import to_binary_image, from_binary_image, decode_stacked
from visual_crypto import extended_vc, stack, load_binary, to_image

# ----------- EDIT HERE (or use command-line arguments) -----------
SECRET_PATH = r""
COVER1_PATH = r""
COVER2_PATH = r""
# ----------- settings -----------
SIZE = 128            # secret is resized to SIZE x SIZE (multiple of 16)
RATIO = 0.5           # compression ratio: fraction of measurements kept (0.25 / 0.5 / 0.75)
BITS = 8              # bits per measurement
COVER_ENHANCE = True  # boost cover contrast so shares resemble the covers better

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data", "outputs")
PKG_DIR = os.path.join(OUT_DIR, "package")
KEY_DIR = os.path.join(HERE, "..", "data", "keys")


# ----------- image helpers -----------
def _gray(src, size):
    img = Image.open(src) if isinstance(src, str) else src
    return img.convert("L").resize(size, Image.LANCZOS)


def _prep_cover(src, size):
    img = _gray(src, size)
    return ImageOps.equalize(img) if COVER_ENHANCE else img


# ----------- SENDER -----------
def sender(secret_src, cover1_src, cover2_src, public_key, ratio=RATIO, size=SIZE, bits=BITS):
    secret = np.array(_gray(secret_src, (size, size))) / 255.0

    seed = lwe.random_seed()                                        # 1
    q, cs_meta = compress(secret, ratio=ratio, seed=seed, bits=bits)  # 2
    payload, info = to_binary_image(q, bits)                        # 3
    side = payload.shape[0]
    cover1 = load_binary(_prep_cover(cover1_src, (side, side)), (side, side))
    cover2 = load_binary(_prep_cover(cover2_src, (side, side)), (side, side))
    share1, share2 = extended_vc(payload, cover1, cover2)           # 4
    ciphertext = lwe.encrypt_seed(public_key, seed)                 # 5

    meta = dict(shape=list(cs_meta["shape"]), block=cs_meta["block"], m=cs_meta["m"],
                bits=cs_meta["bits"], lo=cs_meta["lo"], hi=cs_meta["hi"],
                q_shape=list(info["q_shape"]), n_bits=info["n_bits"])
    return dict(share1=share1, share2=share2, ciphertext=ciphertext, meta=meta, secret=secret)


# ----------- RECEIVER -----------
def receiver(share1, share2, ciphertext, meta, secret_key):
    payload = decode_stacked(stack(share1, share2))                                  # 1
    info = dict(q_shape=tuple(meta["q_shape"]), bits=meta["bits"], n_bits=meta["n_bits"])
    q = from_binary_image(payload, info)                                             # 2
    seed = lwe.decrypt_seed(secret_key, ciphertext)                                  # 3
    cs_meta = dict(shape=tuple(meta["shape"]), block=meta["block"], m=meta["m"],
                   bits=meta["bits"], lo=meta["lo"], hi=meta["hi"])
    return reconstruct(q, cs_meta, seed=seed)                                        # 4


# ----------- save / load the package that travels between sender and receiver -----------
def save_package(pkg, folder=PKG_DIR):
    os.makedirs(folder, exist_ok=True)
    to_image(pkg["share1"]).save(os.path.join(folder, "share1.png"))
    to_image(pkg["share2"]).save(os.path.join(folder, "share2.png"))
    lwe.save_ciphertext(os.path.join(folder, "ciphertext.npz"), pkg["ciphertext"])
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(pkg["meta"], f)


def load_package(folder=PKG_DIR):
    def read_share(name):
        return (np.array(Image.open(os.path.join(folder, name)).convert("L")) == 0).astype(np.uint8)
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    return read_share("share1.png"), read_share("share2.png"), \
        lwe.load_ciphertext(os.path.join(folder, "ciphertext.npz")), meta


def get_keys():
    """Load the receiver's lattice keys, or create and save them the first time."""
    pub, sec = os.path.join(KEY_DIR, "receiver_public.npz"), os.path.join(KEY_DIR, "receiver_secret.npz")
    if os.path.exists(pub) and os.path.exists(sec):
        return lwe.load_public_key(pub), lwe.load_secret_key(sec)
    os.makedirs(KEY_DIR, exist_ok=True)
    public_key, secret_key = lwe.keygen()
    lwe.save_keys(os.path.join(KEY_DIR, "receiver"), public_key, secret_key)
    print(f"(new lattice keys created in {os.path.abspath(KEY_DIR)})")
    return public_key, secret_key


# ----------- run everything on your images -----------
def main():
    args = sys.argv[1:]
    secret_p = r""
    c1_p = r""
    c2_p = r""
    ratio = float(args[3]) if len(args) >= 4 else RATIO
    if not (secret_p and c1_p and c2_p):
        sys.exit("Give three image paths: edit SECRET_PATH / COVER1_PATH / COVER2_PATH at the top "
                 "of pipeline.py, or run: python pipeline.py secret.png cover1.png cover2.png")

    from skimage.metrics import structural_similarity as ssim
    os.makedirs(OUT_DIR, exist_ok=True)
    public_key, secret_key = get_keys()

    # sender side
    t0 = time.time()
    pkg = sender(secret_p, c1_p, c2_p, public_key, ratio=ratio)
    t_send = time.time() - t0
    save_package(pkg)

    # receiver side: works ONLY from the files in the package folder + private key
    t0 = time.time()
    s1, s2, ct, meta = load_package()
    recovered = receiver(s1, s2, ct, meta, secret_key)
    t_recv = time.time() - t0

    secret = pkg["secret"]
    Image.fromarray((secret * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "secret_used.png"))
    Image.fromarray((recovered * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "recovered.png"))
    to_image(stack(s1, s2)).save(os.path.join(OUT_DIR, "stacked_shares.png"))

    # attacker checks
    _, other_key = lwe.keygen()
    wrong_key_img = receiver(s1, s2, ct, meta, other_key)

    print(f"\nSecret {SIZE}x{SIZE} | ratio {ratio} | {BITS}-bit measurements")
    print(f"Each share            : {s1.shape}  (area {s1.size / secret.size:.0f}x the secret)")
    print(f"Lattice ciphertext    : {lwe.ciphertext_size_bytes(ct) / 1024:.0f} KB (sent as separate file)")
    print(f"Sender time / receiver time : {t_send:.1f}s / {t_recv:.1f}s")
    print("\nRESULTS (receiver with correct private key)")
    print(f"  PSNR : {psnr(secret, recovered):.2f} dB")
    print(f"  SSIM : {ssim(secret, recovered, data_range=1.0):.3f}")
    print("ATTACKER with both shares but WRONG lattice key")
    print(f"  PSNR : {psnr(secret, wrong_key_img):.2f} dB  (unrecoverable)")
    print(f"\nPackage (what is sent) : {os.path.abspath(PKG_DIR)}")
    print(f"Result images          : {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()