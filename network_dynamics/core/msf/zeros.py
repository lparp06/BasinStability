"""MSF zero-finding: sign-change detection, noise filtering, linear interpolation.

Public API:
    find_zeros(K, psi, min_sep=1.0, max_zeros=4)
        -> (zeros, brackets, stable_intervals)
"""

import numpy as np


def find_zeros(K, psi, min_sep=1.0, max_zeros=4):
    """
    Locate sign-change zeros of Psi(K) with noise filtering.

    Args:
        K:         1D array of coupling strengths (sorted ascending).
        psi:       1D array of Psi values, same length as K.
        min_sep:   Merge sign-change brackets closer than this distance.
                   Suppresses numerical chatter that produces spurious zeros.
        max_zeros: Maximum number of zeros to return.

    Returns:
        zeros:            List of interpolated K values where Psi crosses zero.
        brackets:         List of (K_left, K_right) bracket pairs.
        stable_intervals: List of (K_lo, K_hi) intervals where Psi < 0.
    """
    finite = np.isfinite(psi)
    sign_change = finite[:-1] & finite[1:] & (psi[:-1] * psi[1:] < 0)
    raw = list(zip(K[:-1][sign_change].tolist(), K[1:][sign_change].tolist()))
    if not raw:
        return [], [], []

    # Merge brackets closer than min_sep (odd count = genuine crossing)
    merged = []
    cs, ce, cc = raw[0][0], raw[0][1], 1
    for lo, hi in raw[1:]:
        if lo - ce < min_sep:
            ce = hi
            cc += 1
        else:
            if cc % 2 == 1:
                merged.append((cs, ce))
            cs, ce, cc = lo, hi, 1
    if cc % 2 == 1:
        merged.append((cs, ce))

    # Linear interpolation to refine each zero location
    zeros = []
    for kl, kr in merged:
        il = int(np.searchsorted(K, kl))
        ir = min(int(np.searchsorted(K, kr)), len(psi) - 1)
        pl, pr = psi[il], psi[ir]
        denom = abs(pl) + abs(pr)
        z = kl + (kr - kl) * abs(pl) / denom if denom > 0 else 0.5*(kl + kr)
        zeros.append(z)

    zeros   = zeros[:max_zeros]
    merged  = merged[:max_zeros]
    stable  = [(merged[i][0], merged[i+1][1])
               for i in range(0, len(merged) - 1, 2)]
    # Odd number of zeros with psi negative at K_max → open-ended stable tail
    if len(merged) % 2 == 1 and np.isfinite(psi[-1]) and psi[-1] < 0:
        stable.append((merged[-1][0], float(K[-1])))
    return zeros, merged, stable
