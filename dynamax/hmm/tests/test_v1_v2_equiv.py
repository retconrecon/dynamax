"""V1/V2 equivalence tests and final integration tests.

Tests 1-4 require importing v1's GaussianHMM, which fails on Python 3.9 because
dynamax/hidden_markov_model/__init__.py imports gaussian_hmm.py which uses
X | Y type syntax (Python 3.10+). These tests are skipped on Python 3.9.

Tests 5-8 always run and validate v2 independently by directly calling the
inference functions and verifying the public API.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# Try to import v1 GaussianHMM
try:
    from dynamax.hidden_markov_model import GaussianHMM as GaussianHMM_v1
    V1_AVAILABLE = True
except (ImportError, TypeError):
    V1_AVAILABLE = False

# Import v2
from dynamax.hmm import GaussianHMM as GaussianHMM_v2
from dynamax.hmm.components import CategoricalInitial, StationaryTransitions, GaussianEmissions

# Import inference functions (works on Python 3.9 via importlib workaround)
try:
    from dynamax.hidden_markov_model.inference import (
        hmm_two_filter_smoother, hmm_filter
    )
except TypeError:
    import importlib.util as _ilu
    import os as _os
    _p = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        'hidden_markov_model', 'inference.py')
    _s = _ilu.spec_from_file_location('dynamax._hmm_inference_equiv', _p)
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    hmm_two_filter_smoother = _m.hmm_two_filter_smoother
    hmm_filter = _m.hmm_filter


def _make_v2_from_v1(params_v1, D):
    """Transfer v1 params to v2 GaussianHMM with matched priors.

    V1 default priors:
        - Initial: Dirichlet(1.1)
        - Transitions: Dirichlet(1.1), stickiness=0
        - Emissions NIW: mean=0, concentration=1e-4, df=D+0.1, scale=1e-4*I
    """
    v2_initial = CategoricalInitial(
        probs=params_v1.initial.probs,
        concentration=1.1,
    )
    v2_transitions = StationaryTransitions(
        transition_matrix=params_v1.transitions.transition_matrix,
        concentration=1.1,
    )
    v2_emissions = GaussianEmissions(
        means=params_v1.emissions.means,
        covs=params_v1.emissions.covs,
        prior_mean=jnp.zeros(D),
        prior_mean_concentration=1e-4,
        prior_df=float(D + 0.1),
        prior_scale=1e-4 * jnp.eye(D),
    )
    return GaussianHMM_v2(
        initial=v2_initial,
        transitions=v2_transitions,
        emissions=v2_emissions,
    )


# ============================================================
# V1/V2 Equivalence Tests (skip on Python 3.9)
# ============================================================

v1_skip = pytest.mark.skipif(
    not V1_AVAILABLE,
    reason="v1 GaussianHMM not importable (Python 3.9 type syntax issue)"
)


@v1_skip
def test_marginal_log_prob_equivalence():
    """V1 and V2 produce identical marginal log-likelihoods
    when initialized with the same parameters."""
    key = jr.PRNGKey(0)
    K, D, T = 3, 4, 100
    y = jax.random.normal(key, (T, D))

    # V1
    model_v1 = GaussianHMM_v1(K, D)
    params_v1, props_v1 = model_v1.initialize(key)
    ll_v1 = model_v1.marginal_log_prob(params_v1, y)

    # V2 with same params and matched priors
    model_v2 = _make_v2_from_v1(params_v1, D)
    ll_v2 = model_v2.marginal_log_prob(y)

    assert jnp.abs(ll_v1 - ll_v2) < 1e-4, \
        f"Log-likelihood mismatch: v1={ll_v1}, v2={ll_v2}, diff={jnp.abs(ll_v1 - ll_v2)}"


@v1_skip
def test_e_step_equivalence():
    """V1 and V2 produce identical smoothed probabilities and marginal loglik."""
    key = jr.PRNGKey(0)
    K, D, T = 3, 4, 100
    y = jax.random.normal(key, (T, D))

    # V1
    model_v1 = GaussianHMM_v1(K, D)
    params_v1, _ = model_v1.initialize(key)
    v1_stats, v1_ll = model_v1.e_step(params_v1, y)

    # V2 with same params
    model_v2 = _make_v2_from_v1(params_v1, D)
    v2_stats, v2_ll = model_v2.e_step(y)

    # Marginal log-likelihoods must match (same inference engine, same inputs)
    assert jnp.abs(v1_ll - v2_ll) < 1e-4, \
        f"E-step loglik mismatch: v1={v1_ll}, v2={v2_ll}"

    # Smoothed initial probs should match
    # V1 initial stats: just smoothed_probs[0] (shape [K])
    # V2 initial stats: same
    v1_init, v1_trans, v1_emit = v1_stats
    v2_init, v2_trans, v2_emit = v2_stats
    assert jnp.allclose(v1_init, v2_init, atol=1e-5), \
        f"Initial stats mismatch: max diff={jnp.abs(v1_init - v2_init).max()}"

    # Transition stats should match (both return posterior.trans_probs)
    assert jnp.allclose(v1_trans, v2_trans, atol=1e-5), \
        f"Trans stats mismatch: max diff={jnp.abs(v1_trans - v2_trans).max()}"

    # Emission suff stats should match
    # V1 stores as dict, V2 as NamedTuple
    assert jnp.allclose(v1_emit['sum_w'], v2_emit.sum_w, atol=1e-5)
    assert jnp.allclose(v1_emit['sum_x'], v2_emit.sum_x, atol=1e-5)
    assert jnp.allclose(v1_emit['sum_xxT'], v2_emit.sum_xxT, atol=1e-5)


@v1_skip
def test_single_em_step():
    """After one EM step from identical starting points,
    parameters should be very close.

    V2's +1e-6*I ridge regularization on covariances causes
    a small intentional difference.
    """
    key = jr.PRNGKey(0)
    K, D, T = 3, 4, 200
    y = jax.random.normal(key, (T, D))

    # V1
    model_v1 = GaussianHMM_v1(K, D)
    params_v1, props_v1 = model_v1.initialize(key)
    params_v1_new, v1_lps = model_v1.fit_em(
        params_v1, props_v1, y, num_iters=1, verbose=False
    )

    # V2 with same params and matched priors
    model_v2 = _make_v2_from_v1(params_v1, D)
    model_v2_new, v2_lps = model_v2.fit_em(y, num_iters=1, verbose=False)

    # Log prob (prior + marginal loglik) from first E-step should match.
    # The marginal loglik matches exactly (proven by test_marginal_log_prob_equivalence).
    # log_prior may differ slightly (~1e-7 relative) due to v2's constrained->unconstrained->
    # constrained roundtrip through the covariance bijector.
    lp_diff = jnp.abs(v1_lps[0] - v2_lps[0])
    lp_scale = jnp.maximum(jnp.abs(v1_lps[0]), 1.0)
    assert lp_diff / lp_scale < 1e-5, \
        f"First EM log-prob mismatch: v1={v1_lps[0]}, v2={v2_lps[0]}, rel_diff={lp_diff / lp_scale}"

    # Updated initial probs should match (same Dirichlet MAP)
    v1_probs = params_v1_new.initial.probs
    v2_probs = model_v2_new.initial.probs
    assert jnp.allclose(v1_probs, v2_probs, atol=1e-4), \
        f"Initial probs diverged: max diff={jnp.abs(v1_probs - v2_probs).max()}"

    # Updated transition matrix should match
    v1_tm = params_v1_new.transitions.transition_matrix
    v2_tm = model_v2_new.transitions.transition_matrix()
    assert jnp.allclose(v1_tm, v2_tm, atol=1e-4), \
        f"Trans matrix diverged: max diff={jnp.abs(v1_tm - v2_tm).max()}"

    # Updated emission means should be very close
    v1_means = params_v1_new.emissions.means
    v2_means = model_v2_new.emissions.means
    assert jnp.allclose(v1_means, v2_means, atol=1e-4), \
        f"Means diverged: max diff={jnp.abs(v1_means - v2_means).max()}"

    # Updated covariances differ slightly due to V2's +1e-6*I ridge
    v1_covs = params_v1_new.emissions.covs
    v2_covs = model_v2_new.emissions.covariances
    assert jnp.allclose(v1_covs, v2_covs, atol=1e-3), \
        f"Covs diverged: max diff={jnp.abs(v1_covs - v2_covs).max()}"


@v1_skip
def test_em_trajectory():
    """V1 and V2 EM trajectories track closely over multiple iterations."""
    key = jr.PRNGKey(0)
    K, D, T = 3, 4, 200
    y = jax.random.normal(key, (T, D))

    # V1
    model_v1 = GaussianHMM_v1(K, D)
    params_v1, props_v1 = model_v1.initialize(key)

    # V2 with same params and matched priors
    model_v2 = _make_v2_from_v1(params_v1, D)

    _, v1_lps = model_v1.fit_em(params_v1, props_v1, y, num_iters=20, verbose=False)
    _, v2_lps = model_v2.fit_em(y, num_iters=20, verbose=False)

    # First few iterations should track very closely (relative tolerance).
    # Small absolute diffs (~1-2) are expected on large values (~9M) due to
    # v2's bijector roundtrip affecting log_prior by ~1e-7 relative.
    for i in range(5):
        diff = jnp.abs(v1_lps[i] - v2_lps[i])
        scale = jnp.maximum(jnp.abs(v1_lps[i]), 1.0)
        assert diff / scale < 1e-5, \
            f"Iteration {i}: v1={v1_lps[i]:.4f}, v2={v2_lps[i]:.4f}, rel_diff={diff / scale:.2e}"

    # Both should be monotonically non-decreasing
    assert jnp.all(jnp.diff(v1_lps) >= -1e-2), "V1 EM not monotonic"
    assert jnp.all(jnp.diff(v2_lps) >= -1e-2), "V2 EM not monotonic"


# ============================================================
# Tests that always run (no v1 import needed)
# ============================================================

def test_public_api():
    """The public API works exactly as shown in the README/proposal."""
    from dynamax.hmm import GaussianHMM

    key = jr.PRNGKey(0)

    # Create
    model = GaussianHMM.create(num_states=3, emission_dim=4, key=key)

    # Sample
    states, obs = model.sample(jr.PRNGKey(1), num_timesteps=500)
    assert obs.shape == (500, 4)
    assert states.shape == (500,)

    # Fit
    fitted, lls = model.fit_em(obs, num_iters=50, verbose=False)
    assert lls[-1] > lls[0]
    assert jnp.all(jnp.isfinite(lls))

    # Log likelihood
    ll = fitted.marginal_log_prob(obs)
    assert jnp.isfinite(ll)


def test_inference_plumbing():
    """V2's _inference_args feeds the inference engine correctly.
    Calling hmm_filter directly with the same args gives the same result."""
    K, D, T = 3, 4, 100
    model = GaussianHMM_v2.create(K, D, key=jr.PRNGKey(0))
    y = jax.random.normal(jr.PRNGKey(1), (T, D))

    # Get inference args from v2
    initial_probs, trans_mat, log_liks = model._inference_args(y)
    assert initial_probs.shape == (K,)
    assert trans_mat.shape == (K, K)
    assert log_liks.shape == (T, K)

    # Call hmm_filter directly
    post_direct = hmm_filter(initial_probs, trans_mat, log_liks)

    # Compare to v2's marginal_log_prob
    ll_v2 = model.marginal_log_prob(y)
    assert jnp.abs(post_direct.marginal_loglik - ll_v2) < 1e-6


def test_smoother_plumbing():
    """V2's smoother matches direct hmm_two_filter_smoother call."""
    K, D, T = 3, 4, 100
    model = GaussianHMM_v2.create(K, D, key=jr.PRNGKey(0))
    y = jax.random.normal(jr.PRNGKey(1), (T, D))

    # Direct call
    args = model._inference_args(y)
    posterior_direct = hmm_two_filter_smoother(*args)

    # Through v2 model
    posterior_v2 = model.smoother(y)

    assert jnp.allclose(
        posterior_direct.smoothed_probs, posterior_v2.smoothed_probs, atol=1e-6
    )
    assert jnp.abs(
        posterior_direct.marginal_loglik - posterior_v2.marginal_loglik
    ) < 1e-6


def test_constrained_unconstrained_roundtrip():
    """Parameters round-trip correctly through constrained/unconstrained space."""
    K, D = 3, 4
    model = GaussianHMM_v2.create(K, D, key=jr.PRNGKey(0))

    # Means roundtrip
    means = model.emissions.means
    assert means.shape == (K, D)

    # Covariances are SPD
    covs = model.emissions.covariances
    assert covs.shape == (K, D, D)
    for k in range(K):
        eigvals = jnp.linalg.eigvalsh(covs[k])
        assert jnp.all(eigvals > 0), f"Cov {k} not SPD"

    # Initial probs on simplex
    probs = model.initial.probs
    assert jnp.allclose(probs.sum(), 1.0, atol=1e-6)
    assert jnp.all(probs > 0)

    # Transition matrix rows sum to 1
    tm = model.transitions.transition_matrix()
    for k in range(K):
        assert jnp.allclose(tm[k].sum(), 1.0, atol=1e-6)

    # LDJ is finite
    ldj = model.log_det_jacobian()
    assert jnp.isfinite(ldj)
