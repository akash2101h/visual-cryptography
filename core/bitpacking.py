"""
Bit-packing module: integers  <->  bits  <->  binary image.

WHY
  Compressive sensing gives integers (the quantized measurements).
  Visual cryptography needs a black-and-white image.
  So: write each integer as `bits` bits (MSB first) and lay all bits out as pixels.
  Convention (same as visual_crypto.py): 1 = BLACK, 0 = WHITE.

The receiver needs three public (non-secret) numbers to unpack: q_shape, bits, and
nothing else. They are carried in `meta`, not hidden.
"""
import numpy as np


def to_binary_image(q, bits):
    """
    q    : integer array (any shape), values in 0 .. 2^bits - 1
    returns (img, info)
        img  : square 0/1 array (padded with zeros at the end)
        info : dict(q_shape, bits, n_bits)  -> needed to unpack
    """
    v = q.astype(np.uint32).ravel()
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bit_array = ((v[:, None] >> shifts) & 1).astype(np.uint8).ravel()  # MSB first
    n_bits = bit_array.size
    side = int(np.ceil(np.sqrt(n_bits)))
    padded = np.zeros(side * side, dtype=np.uint8)
    padded[:n_bits] = bit_array
    info = dict(q_shape=q.shape, bits=bits, n_bits=n_bits)
    return padded.reshape(side, side), info


def from_binary_image(img, info):
    """Exact inverse of to_binary_image."""
    bits = info["bits"]
    flat = img.ravel()[: info["n_bits"]].astype(np.uint32)
    groups = flat.reshape(-1, bits)
    weights = 2 ** np.arange(bits - 1, -1, -1, dtype=np.uint32)
    q = (groups * weights).sum(axis=1)
    return q.reshape(info["q_shape"]).astype(np.uint16)


def decode_stacked(stacked):
    """
    Digital decoding of the stacked shares (2x2 sub-pixels per payload bit).
    A payload BLACK pixel -> all 4 sub-pixels black.
    A payload WHITE pixel -> at most 3 black (basic VC: 2, meaningful VC: 2 or 3).
    So 'block fully black' <=> bit = 1. This is exact, no contrast loss in digital mode.
    """
    H, W = stacked.shape[0] // 2, stacked.shape[1] // 2
    block_sum = stacked.reshape(H, 2, W, 2).sum(axis=(1, 3))
    return (block_sum == 4).astype(np.uint8)


# ---------------- self tests ----------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Test 1: pure round trip for several bit depths
    for bits in (4, 6, 8):
        q = rng.integers(0, 2 ** bits, size=(64, 128)).astype(np.uint16)
        img, info = to_binary_image(q, bits)
        back = from_binary_image(img, info)
        print(f"Test 1 round trip, {bits}-bit: image {img.shape}, exact = {np.array_equal(q, back)}")

    # Test 2: full path through visual cryptography + compressive sensing
    from skimage import data
    from skimage.transform import resize
    from compressive_sensing import compress, reconstruct, psnr
    from visual_crypto import extended_vc, basic_vc, stack, load_binary

    secret = resize(data.camera() / 255.0, (128, 128), anti_aliasing=True)
    q, meta = compress(secret, ratio=0.5, bits=8)
    payload, info = to_binary_image(q, 8)
    print(f"\nTest 2 full chain: payload image {payload.shape} ({info['n_bits']} bits)")

    from PIL import Image
    size = payload.shape[::-1]

    def as_cover(arr):
        gray = (resize(arr / 255.0, payload.shape, anti_aliasing=True) * 255).astype(np.uint8)
        return load_binary(Image.fromarray(gray), size)

    cover1 = as_cover(data.moon())
    cover2 = as_cover(data.coins())

    for name, (s1, s2) in [("basic VC   ", basic_vc(payload)),
                           ("meaningful ", extended_vc(payload, cover1, cover2))]:
        decoded = decode_stacked(stack(s1, s2))
        q_back = from_binary_image(decoded, info)
        rec = reconstruct(q_back, meta)
        print(f"  {name}: share {s1.shape}, payload recovered exactly = "
              f"{np.array_equal(decoded, payload)}, q exact = {np.array_equal(q, q_back)}, "
              f"final PSNR = {psnr(secret, rec):.2f} dB")