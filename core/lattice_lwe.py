"""
Lattice cryptography module: simplified Regev LWE (Learning With Errors) encryption.

PURPOSE IN THE PROJECT
  The compressive-sensing matrix Phi is generated from a secret SEED (128 bits).
  Without the seed, the measurements cannot be turned back into the image.
  This module encrypts that SEED with LWE, so only the owner of the lattice private key
  can recover it.  (We encrypt only the small seed, not the image: LWE ciphertexts are large.)

HOW REGEV LWE WORKS (one bit at a time)
  Parameters : dimension n, modulus q, number of samples m, small errors
  Key generation:
      secret key  s  : random vector in Z_q^n
      public key (A, b):  A random m x n matrix,   b = A.s + e  (mod q),   e = small noise
  Encrypt bit mu:
      pick random 0/1 vector r (length m)
      u = A^T r (mod q)           v = b.r + mu * floor(q/2) (mod q)
  Decrypt:
      d = v - s.u (mod q)  =  e.r + mu*floor(q/2)
      e.r is small, so d is near 0 -> bit 0, or near q/2 -> bit 1.
  Security idea: recovering s from (A, b) means separating A.s from the noise e. That is the
  LWE problem, believed hard even for quantum computers.

!! EDUCATIONAL IMPLEMENTATION. Parameters follow common lattice sizes (n=256, q=3329) but this
!! code is not audited or constant-time. A real system should use ML-KEM (Kyber).
"""
import os
import numpy as np

N = 256          # lattice dimension
Q = 3329         # modulus
M = 512          # number of samples in the public key
ETA = 2          # noise: centred binomial with parameter 2  -> values in [-2, 2]
SEED_BITS = 128


def _rng():
    """Fresh randomness from the operating system."""
    return np.random.default_rng(int.from_bytes(os.urandom(16), "big"))


def _noise(rng, size):
    """Centred binomial noise: small integers in [-ETA, ETA]."""
    return rng.integers(0, 2, size=(size, 2 * ETA)).sum(axis=1) - ETA


# ---------------- key generation ----------------
def keygen():
    """Returns (public_key, secret_key). public_key = (A, b), secret_key = s."""
    rng = _rng()
    A = rng.integers(0, Q, size=(M, N), dtype=np.int64)
    s = rng.integers(0, Q, size=N, dtype=np.int64)
    e = _noise(rng, M).astype(np.int64)
    b = (A @ s + e) % Q
    return (A, b), s


# ---------------- seed <-> bits ----------------
def random_seed():
    """A fresh random 128-bit seed (as a Python int) for compressive sensing."""
    return int.from_bytes(os.urandom(SEED_BITS // 8), "big")


def _seed_to_bits(seed):
    return np.array([(seed >> (SEED_BITS - 1 - i)) & 1 for i in range(SEED_BITS)], dtype=np.int64)


def _bits_to_seed(bits):
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


# ---------------- encrypt / decrypt ----------------
def encrypt_seed(public_key, seed):
    """Encrypt a 128-bit seed. Returns ciphertext (U, v): U is (128 x N), v is (128,)."""
    A, b = public_key
    rng = _rng()
    mu = _seed_to_bits(seed)
    R = rng.integers(0, 2, size=(SEED_BITS, M), dtype=np.int64)   # one random 0/1 vector per bit
    U = (R @ A) % Q
    v = (R @ b + mu * (Q // 2)) % Q
    return U, v


def decrypt_seed(secret_key, ciphertext):
    """Recover the seed. With the wrong secret key the result is a wrong (random-looking) seed."""
    U, v = ciphertext
    d = (v - U @ secret_key) % Q
    bits = ((d > Q // 4) & (d < 3 * Q // 4)).astype(np.int64)    # closer to q/2 than to 0
    return _bits_to_seed(bits)


def ciphertext_size_bytes(ciphertext):
    """Approximate size if each number is stored in 12 bits (q < 4096)."""
    U, v = ciphertext
    return (U.size + v.size) * 12 // 8


def save_ciphertext(path, ciphertext):
    np.savez_compressed(path, U=ciphertext[0], v=ciphertext[1])


def load_ciphertext(path):
    z = np.load(path)
    return z["U"], z["v"]


def save_keys(path_prefix, public_key, secret_key):
    np.savez_compressed(path_prefix + "_public.npz", A=public_key[0], b=public_key[1])
    np.savez_compressed(path_prefix + "_secret.npz", s=secret_key)


def load_public_key(path):
    z = np.load(path)
    return z["A"], z["b"]


def load_secret_key(path):
    return np.load(path)["s"]


# ---------------- self tests ----------------
if __name__ == "__main__":
    import time

    pk, sk = keygen()
    print(f"Parameters: n={N}, q={Q}, m={M}, noise in [-{ETA},{ETA}], seed = {SEED_BITS} bits\n")

    # Test 1: correctness over many random seeds
    trials, ok = 300, 0
    t0 = time.time()
    for _ in range(trials):
        seed = random_seed()
        ok += decrypt_seed(sk, encrypt_seed(pk, seed)) == seed
    print(f"Test 1 correctness: {ok}/{trials} seeds decrypted exactly  ({time.time() - t0:.1f}s)")

    # Test 2: wrong key -> garbage
    seed = random_seed()
    ct = encrypt_seed(pk, seed)
    _, other_sk = keygen()
    wrong = decrypt_seed(other_sk, ct)
    diff_bits = bin(seed ^ wrong).count("1")
    print(f"Test 2 wrong key  : seed recovered = {wrong == seed}, "
          f"{diff_bits}/{SEED_BITS} bits differ (about half is expected = random guess)")

    # Test 3: same seed encrypted twice gives different ciphertext
    ct2 = encrypt_seed(pk, seed)
    print(f"Test 3 randomized : two encryptions of the same seed identical = "
          f"{np.array_equal(ct[0], ct2[0])}  (must be False)")

    # Test 4: cost
    print(f"Test 4 size       : ciphertext about {ciphertext_size_bytes(ct) / 1024:.0f} KB "
          f"for a 16-byte seed (LWE is large; that is why we encrypt only the seed)")

    # Test 5: connect to compressive sensing - right key vs wrong key
    from skimage import data
    from skimage.transform import resize
    from compressive_sensing import compress, reconstruct, psnr

    img = resize(data.camera() / 255.0, (128, 128), anti_aliasing=True)
    seed = random_seed()
    q, meta = compress(img, ratio=0.5, seed=seed, bits=8)
    ct = encrypt_seed(pk, seed)
    good = reconstruct(q, meta, seed=decrypt_seed(sk, ct))
    bad = reconstruct(q, meta, seed=decrypt_seed(other_sk, ct))
    print(f"\nTest 5 chain with compressive sensing (same measurements):")
    print(f"  correct lattice key : PSNR = {psnr(img, good):.2f} dB")
    print(f"  wrong lattice key   : PSNR = {psnr(img, bad):.2f} dB  (image is not recoverable)")