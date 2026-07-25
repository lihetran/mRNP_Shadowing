'''
July 24, 2026 LT (written with Claude)

Regression tests for the explicit-duration (semi-Markov) HMM machinery in
runHMMPerGene.py / trainHMMPerGene.py: _forward_backward_hsmm,
_duration_to_hazard, _duration_pmf_default, classify_positions_hmm3,
train_hsmm_durations.

Run directly: python3 test_hsmm.py
(plain assert-based, no pytest dependency, matching the rest of this
project's scripts)

Why this exists: the first version of _forward_backward_hsmm used hand-
derived segment-indexed forward-backward recursions that had a real bug --
P_B posteriors escaping [0,1] (observed up to ~395), overall log-likelihood
off by ~9 nats -- caught only by cross-checking against an independently
written brute-force reference. That reference (expanding the duration
distribution into extra Markov states with position-specific hazard rates)
turned out to be an EXACT equivalent of the explicit-duration model, not
just an approximation, so it became the actual implementation. These tests
keep both the cross-check and the "does EM actually recover known ground
truth" demonstration around so future changes to this code don't regress
silently the same way.
'''

import numpy as np

import runHMMPerGene as R
import trainHMMPerGene as T


def _independent_reference_forward_backward(coords, eA, eB, pi_A, a_AB, D_B):
    """
    Deliberately separate implementation of the same expanded-state
    construction _forward_backward_hsmm uses, written independently so a
    future refactor of the production function (e.g. for speed) has
    something to be checked against. Intentionally simple/unoptimized --
    do not "clean this up" to share code with _forward_backward_hsmm, that
    would defeat the point of having an independent check.
    """
    Dmax = len(D_B)
    tail = np.cumsum(D_B[::-1])[::-1]
    hazard = np.array([D_B[i] / tail[i] if tail[i] > 1e-300 else 1.0
                       for i in range(Dmax)])
    hazard[-1] = 1.0

    S = 1 + Dmax
    T_mat = np.zeros((S, S))
    T_mat[0, 0] = 1 - a_AB
    T_mat[0, 1] = a_AB
    for d in range(1, Dmax):
        T_mat[d, d + 1] = 1 - hazard[d - 1]
        T_mat[d, 0]     = hazard[d - 1]
    T_mat[Dmax, 0] = 1.0

    pi = np.zeros(S)
    pi[0] = pi_A
    pi[1] = 1.0 - pi_A

    L = int(coords[-1] - coords[0]) + 1
    site_at = {int(c - coords[0]): i for i, c in enumerate(coords)}

    def emis_vec(t):
        e = np.ones(S)
        i = site_at.get(t)
        if i is not None:
            e[0]  = eA[i]
            e[1:] = eB[i]
        return e

    alpha = np.zeros((L, S)); c = np.zeros(L)
    v = pi * emis_vec(0); c[0] = v.sum(); alpha[0] = v / c[0]
    for t in range(1, L):
        v = (alpha[t - 1] @ T_mat) * emis_vec(t)
        c[t] = v.sum(); alpha[t] = v / c[t]

    beta = np.zeros((L, S)); beta[-1] = 1.0
    for t in range(L - 2, -1, -1):
        beta[t] = (T_mat @ (emis_vec(t + 1) * beta[t + 1])) / c[t + 1]

    post = alpha * beta
    post /= post.sum(axis=1, keepdims=True)

    post_B = np.array([post[int(c_ - coords[0]), 1:].sum() for c_ in coords])
    loglik = float(np.sum(np.log(c)))
    return post_B, loglik


def make_synthetic_read(rng, gene_len=300, site_spacing=3,
                        pA_true=0.6, pB_true=0.03,
                        footprint_lo=120, footprint_hi=150):
    """One read with a single injected protected region -- for quick,
    deterministic sanity checks (not a full generative simulation)."""
    coords = np.arange(0, gene_len, site_spacing, dtype=float)
    bits = np.array([1 if rng.random() < (pB_true if footprint_lo <= c <= footprint_hi
                                          else pA_true) else 0
                     for c in coords])
    eA = np.array([pA_true if b else 1 - pA_true for b in bits])
    eB = np.array([pB_true if b else 1 - pB_true for b in bits])
    return coords, eA, eB


def simulate_generative_read(rng, gene_len, site_spacing, pi_A, a_AB_true,
                             D_B_true, pA_true, pB_true):
    """Full nt-level semi-Markov simulation -- for testing whether EM
    recovers known ground-truth parameters, independent of any of the
    forward-backward code being tested."""
    state = 0 if rng.random() < pi_A else 1
    pos = 0
    Dmax = len(D_B_true)
    state_at = np.zeros(gene_len, dtype=int)
    while pos < gene_len:
        if state == 0:
            state_at[pos] = 0
            if rng.random() < a_AB_true:
                d = rng.choice(np.arange(1, Dmax + 1), p=D_B_true)
                for j in range(d):
                    if pos + 1 + j < gene_len:
                        state_at[pos + 1 + j] = 1
                pos += 1 + d
            else:
                pos += 1
        else:
            pos += 1

    coords = np.arange(0, gene_len, site_spacing)
    bits = [1 if rng.random() < (pB_true if state_at[c] == 1 else pA_true) else 0
            for c in coords]
    eA = np.array([pA_true if b else 1 - pA_true for b in bits])
    eB = np.array([pB_true if b else 1 - pB_true for b in bits])
    return coords.astype(float), eA, eB


def test_probabilities_bounded():
    """post_B must always be a valid probability -- the exact invariant the
    original buggy segment-math implementation violated (values up to 395),
    checked across several random single-read cases, not just one."""
    rng = np.random.default_rng(0)
    D_B = R._duration_pmf_default(mean_nt=30, sd_nt=6, dmax=120)
    for trial in range(10):
        lo = int(rng.integers(0, 250))
        hi = lo + 30
        coords, eA, eB = make_synthetic_read(rng, footprint_lo=lo, footprint_hi=hi)

        post_B, expected_len, loglik, extra = R._forward_backward_hsmm(
            coords, eA, eB, pi_A=0.8, a_AB=0.01, D_B=D_B)

        assert not np.isnan(post_B).any(), f"trial {trial}: NaN in post_B"
        assert post_B.min() >= -1e-9, f"trial {trial}: post_B below 0: {post_B.min()}"
        assert post_B.max() <= 1 + 1e-9, f"trial {trial}: post_B above 1: {post_B.max()}"
        assert np.isfinite(loglik), f"trial {trial}: non-finite loglik"
    print("test_probabilities_bounded: PASS")


def test_matches_independent_reference():
    """_forward_backward_hsmm must agree with a separately-written
    implementation of the same construction, across several random cases --
    guards against a future refactor (e.g. for speed) silently breaking
    correctness."""
    rng = np.random.default_rng(1)
    D_B = R._duration_pmf_default(mean_nt=25, sd_nt=8, dmax=100)
    for trial in range(5):
        lo = int(rng.integers(0, 250))
        coords, eA, eB = make_synthetic_read(rng, footprint_lo=lo, footprint_hi=lo + 25)

        post_B, _expected_len, loglik, _extra = R._forward_backward_hsmm(
            coords, eA, eB, pi_A=0.75, a_AB=0.012, D_B=D_B)
        ref_post_B, ref_loglik = _independent_reference_forward_backward(
            coords, eA, eB, pi_A=0.75, a_AB=0.012, D_B=D_B)

        assert np.allclose(post_B, ref_post_B, atol=1e-8), \
            f"trial {trial}: post_B mismatch, max diff {np.max(np.abs(post_B - ref_post_B))}"
        assert abs(loglik - ref_loglik) < 1e-6, \
            f"trial {trial}: loglik mismatch {loglik} vs {ref_loglik}"
    print("test_matches_independent_reference: PASS")


def test_edge_cases_no_crash():
    D_B = R._duration_pmf_default(mean_nt=30, sd_nt=6, dmax=120)
    post_B, _, loglik, _ = R._forward_backward_hsmm(
        np.array([50.0]), np.array([0.9]), np.array([0.1]), 0.8, 0.01, D_B)
    assert post_B.shape == (1,) and np.isfinite(loglik)

    post_B, _, loglik, _ = R._forward_backward_hsmm(
        np.array([0.0, 500.0]), np.array([0.9, 0.1]), np.array([0.1, 0.9]),
        0.8, 0.01, D_B)
    assert post_B.shape == (2,) and np.isfinite(loglik)
    print("test_edge_cases_no_crash: PASS")


def test_recovers_known_footprint_and_rate():
    """The real payoff check: simulate the actual generative process with a
    KNOWN footprint-length distribution and entry rate, run Baum-Welch, and
    confirm it recovers something close to ground truth -- plus check the
    EM log-likelihood-improves-every-iteration guarantee holds. A violation
    of that guarantee is a very strong signal of a math bug, independent of
    whether the fitted answer happens to look reasonable."""
    rng = np.random.default_rng(42)
    true_mean_nt, true_sd_nt, true_a_AB = 30.0, 5.0, 0.008
    D_B_true = R._duration_pmf_default(mean_nt=true_mean_nt, sd_nt=true_sd_nt, dmax=120)
    pi_A, pA_true, pB_true = 0.85, 0.55, 0.04

    reads = [simulate_generative_read(rng, gene_len=900, site_spacing=3,
                                      pi_A=pi_A, a_AB_true=true_a_AB,
                                      D_B_true=D_B_true, pA_true=pA_true,
                                      pB_true=pB_true)
              for _ in range(150)]

    D_B_fit, a_AB_fit, history = T.train_hsmm_durations(
        reads, pi_A, max_iters=15, tol=1e-5, verbose=False)

    diffs = np.diff(history)
    assert (diffs >= -1e-6).all(), \
        f"log-likelihood decreased at some iteration: {history}"

    d = np.arange(1, len(D_B_fit) + 1)
    fit_mean = (d * D_B_fit).sum()
    fit_sd   = (((d - fit_mean) ** 2 * D_B_fit).sum()) ** 0.5

    assert abs(fit_mean - true_mean_nt) < 5.0, \
        f"fitted mean {fit_mean:.2f}nt too far from true {true_mean_nt}nt"
    assert abs(fit_sd - true_sd_nt) < 5.0, \
        f"fitted sd {fit_sd:.2f}nt too far from true {true_sd_nt}nt"
    assert abs(a_AB_fit - true_a_AB) / true_a_AB < 0.5, \
        f"fitted a_AB {a_AB_fit:.5f} too far from true {true_a_AB}"

    print(f"test_recovers_known_footprint_and_rate: PASS "
          f"(fitted mean={fit_mean:.2f}nt sd={fit_sd:.2f}nt a_AB={a_AB_fit:.5f}; "
          f"true mean={true_mean_nt}nt sd={true_sd_nt}nt a_AB={true_a_AB})")


if __name__ == "__main__":
    test_probabilities_bounded()
    test_matches_independent_reference()
    test_edge_cases_no_crash()
    test_recovers_known_footprint_and_rate()
    print("\nAll HSMM tests passed.")
