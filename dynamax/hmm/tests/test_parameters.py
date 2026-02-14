"""Tests for ConstrainedParameter."""
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb

from dynamax.hmm.parameters import ConstrainedParameter


def test_unconstrained_param_roundtrip():
    """Identity bijector: value == unconstrained."""
    x = jnp.array([1.0, 2.0, 3.0])
    cp = ConstrainedParameter.unconstrained_param(x)
    assert jnp.allclose(cp.value, x)
    assert jnp.allclose(cp.unconstrained, x)


def test_softplus_bijector():
    """Softplus: constrained values are positive."""
    x = jnp.array([-2.0, 0.0, 2.0])  # unconstrained
    cp = ConstrainedParameter(
        _unconstrained=x,
        bijector=tfb.Softplus(),
        trainable=True,
    )
    assert jnp.all(cp.value > 0)


def test_from_constrained_roundtrip():
    """from_constrained -> value recovers original."""
    original = jnp.array([0.1, 0.3, 0.6])  # simplex
    cp = ConstrainedParameter.from_constrained(
        original, tfb.SoftmaxCentered(), trainable=True
    )
    assert jnp.allclose(cp.value, original, atol=1e-5)


def test_softmax_centered_shapes():
    """SoftmaxCentered: unconstrained is K-1, constrained is K."""
    K = 5
    probs = jnp.ones(K) / K  # uniform simplex
    cp = ConstrainedParameter.from_constrained(
        probs, tfb.SoftmaxCentered(), trainable=True
    )
    assert cp.unconstrained.shape == (K - 1,)
    assert cp.value.shape == (K,)
    assert jnp.allclose(cp.value.sum(), 1.0, atol=1e-6)


def test_freeze_unfreeze():
    """Freeze stops gradients, unfreeze restores them."""
    x = jnp.array([1.0, 2.0])
    cp = ConstrainedParameter.unconstrained_param(x, trainable=True)
    assert cp.trainable is True

    frozen = cp.freeze()
    assert frozen.trainable is False
    assert jnp.allclose(frozen.value, cp.value)

    unfrozen = frozen.unfreeze()
    assert unfrozen.trainable is True


def test_ldj_zero_when_frozen():
    """Frozen parameters contribute zero to log-det-Jacobian."""
    x = jnp.array([1.0, 2.0])
    cp = ConstrainedParameter(
        _unconstrained=x,
        bijector=tfb.Softplus(),
        trainable=False,
    )
    assert cp.log_det_jacobian() == 0.0


def test_ldj_finite_trainable():
    """Trainable parameters have finite log-det-Jacobian."""
    x = jnp.array([1.0, 2.0])
    cp = ConstrainedParameter(
        _unconstrained=x,
        bijector=tfb.Softplus(),
        trainable=True,
    )
    ldj = cp.log_det_jacobian()
    assert jnp.isfinite(ldj)
    assert isinstance(ldj, jnp.ndarray)  # JAX array, not Python float


def test_stop_gradient_when_frozen():
    """Frozen parameter value has zero gradient."""
    x = jnp.array([1.0, 2.0])
    cp = ConstrainedParameter(
        _unconstrained=x,
        bijector=tfb.Softplus(),
        trainable=False,
    )

    def f(cp):
        return cp.value.sum()

    grad = jax.grad(f)(cp)
    assert jnp.allclose(grad._unconstrained, 0.0)


def test_pytree_behavior():
    """ConstrainedParameter is a valid JAX pytree."""
    cp = ConstrainedParameter.unconstrained_param(jnp.array([1.0, 2.0]))
    leaves, treedef = jax.tree_util.tree_flatten(cp)
    # _unconstrained should be the only leaf
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], jnp.array([1.0, 2.0]))
    # Reconstruct
    cp2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(cp2.value, cp.value)


def test_softmax_centered_ldj():
    """SoftmaxCentered log-det-Jacobian is finite."""
    K = 4
    probs = jnp.ones(K) / K
    cp = ConstrainedParameter.from_constrained(
        probs, tfb.SoftmaxCentered(), trainable=True
    )
    ldj = cp.log_det_jacobian()
    assert jnp.isfinite(ldj)
