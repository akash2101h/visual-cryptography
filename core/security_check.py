"""
Security check for the 2x2 (OR) share construction.

Question: if an attacker holds ONE share and knows BOTH cover images exactly, does the share
reveal anything about the payload pixel?

Method: for every pair of cover pixels (c1, c2) and for each share separately, compare the
probability distribution of that share's 2x2 pattern when the payload pixel is white with the
distribution when it is black. If they are identical, the share is statistically independent of the
payload. This holds only if the pattern is chosen with true randomness (visual_crypto.py uses the
operating system's secure generator via the secrets module).

Run:  python security_check.py
"""
from collections import Counter
from visual_crypto import TABLE


def single_share_independent():
    for which, name in ((0, "share 1"), (1, "share 2")):
        for c1 in (0, 1):
            for c2 in (0, 1):
                dist = []
                for s in (0, 1):                              # payload pixel white / black
                    cnt = Counter(tuple(pair[which].ravel()) for pair in TABLE[(s, c1, c2)])
                    tot = sum(cnt.values())
                    dist.append({k: v / tot for k, v in cnt.items()})
                same = dist[0].keys() == dist[1].keys() and all(
                    abs(dist[0][k] - dist[1][k]) < 1e-12 for k in dist[0])
                if not same:
                    return False, f"{name} depends on the payload for covers ({c1},{c2})"
    return True, "every share pattern has the same distribution for white and black payload pixels"


if __name__ == "__main__":
    ok, msg = single_share_independent()
    print("Single share independent of payload:", ok)
    print(msg)