""":class:`AbstractUnwrappable` objects and utilities.

These are placeholder values for specifying custom behaviour for nodes in a pytree,
applied using :func:`unwrap`.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn import softplus
from jax.tree_util import tree_leaves
from jaxtyping import Array, PyTree

from paramax.utils import inv_softplus

T = TypeVar("T")


class AbstractUnwrappable(eqx.Module, Generic[T]):
    """An abstract class representing an unwrappable object.

    Unwrappables replace PyTree nodes, applying custom behavior upon unwrapping.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        pass


def unwrap(tree: PyTree):
    """Map across a PyTree and unwrap all :class:`AbstractUnwrappable` nodes.

    This leaves all other nodes unchanged. If nested, the innermost
    ``AbstractUnwrappable`` nodes are unwrapped first.

    Example:
        Enforcing positivity.

        .. doctest::

            >>> import paramax
            >>> import jax.numpy as jnp
            >>> params = paramax.Parameterize(jnp.exp, jnp.zeros(3))
            >>> paramax.unwrap(("abc", 1, params))
            ('abc', 1, Array([1., 1., 1.], dtype=float32))
    """

    def _unwrap(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, AbstractUnwrappable):
                # Unwrap subnodes, then itself
                return _unwrap(leaf, include_self=False).unwrap()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, AbstractUnwrappable)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree_util.tree_map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _unwrap(tree, include_self=True)


class Parameterize(AbstractUnwrappable[T]):
    """Unwrap an object by calling fn with args and kwargs.

    All of fn, args and kwargs may contain trainable parameters.

    .. note::

        Unwrapping typically occurs after model initialization. Therefore, if the
        ``Parameterize`` object may be created in a vectorized context, we recommend
        ensuring that ``fn`` still unwraps correctly, e.g. by supporting broadcasting.

    Example:
        .. doctest::

            >>> from paramax.wrappers import Parameterize, unwrap
            >>> import jax.numpy as jnp
            >>> positive = Parameterize(jnp.exp, jnp.zeros(3))
            >>> unwrap(positive)  # Aplies exp on unwrapping
            Array([1., 1., 1.], dtype=float32)

    Args:
        fn: Callable to call with args, and kwargs.
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)


def non_trainable(tree: PyTree):
    """Freezes parameters by wrapping inexact array leaves with :class:`NonTrainable`.

    .. note::

        Regularization is likely to apply before unwrapping. To avoid regularization
        impacting non-trainable parameters, they should be filtered out,
        for example using:

        .. code-block:: python

            >>> eqx.partition(
            ...     ...,
            ...     is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
            ... )


    Wrapping the arrays in a model rather than the entire tree is often preferable,
    allowing easier access to attributes compared to wrapping the entire tree.

    Args:
        tree: The pytree.
    """

    def _map_fn(leaf):
        return NonTrainable(leaf) if eqx.is_inexact_array(leaf) else leaf

    return jax.tree_util.tree_map(
        f=_map_fn,
        tree=tree,
        is_leaf=lambda x: isinstance(x, NonTrainable),
    )


class NonTrainable(AbstractUnwrappable[T]):
    """Applies stop gradient to all arraylike leaves before unwrapping.

    See also :func:`non_trainable`, which is probably a generally prefereable way to
    achieve similar behaviour, which wraps the arraylike leaves directly, rather than
    the tree. Useful to mark pytrees (arrays, submodules, etc) as frozen/non-trainable.
    Note that the underlying parameters may still be impacted by regularization,
    so it is generally advised to use this as a suggestively named class
    for filtering parameters.
    """

    tree: T

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(lax.stop_gradient(differentiable), static)


class WeightNormalization(AbstractUnwrappable[Array]):
    """Applies weight normalization (https://arxiv.org/abs/1602.07868).

    Args:
        weight: The (possibly wrapped) weight matrix.
    """

    weight: Array | AbstractUnwrappable[Array]
    scale: Array | AbstractUnwrappable[Array]

    def __init__(self, weight: Array | AbstractUnwrappable[Array]):
        self.weight = weight
        scale_init = 1 / jnp.linalg.norm(unwrap(weight), axis=-1, keepdims=True)
        self.scale = Parameterize(softplus, inv_softplus(scale_init))

    def unwrap(self) -> Array:
        weight_norms = jnp.linalg.norm(self.weight, axis=-1, keepdims=True)
        return self.scale * self.weight / weight_norms


class BiasOrdering(AbstractUnwrappable[Array]):
    """Applies bias ordering (http://bayesiandeeplearning.org/2017/papers/15.pdf).

    Args:
        bias: The (possibly wrapped) bias vector.
    """

    bias: Array | AbstractUnwrappable[Array]

    def unwrap(self) -> Array:
        first, log_diffs = self.bias[..., :1], self.bias[..., 1:]
        first_and_diffs = jnp.concatenate([first, jnp.exp(log_diffs)], axis=-1)
        return jnp.cumsum(first_and_diffs, axis=-1)


def contains_unwrappables(pytree):
    """Check if a pytree contains unwrappables."""

    def _is_unwrappable(leaf):
        return isinstance(leaf, AbstractUnwrappable)

    leaves = tree_leaves(pytree, is_leaf=_is_unwrappable)
    return any(_is_unwrappable(leaf) for leaf in leaves)
