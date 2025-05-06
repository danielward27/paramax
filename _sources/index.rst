Paramax
===========

A small package for applying parameterizations and constraints to nodes in JAX
PyTrees.


Installation
------------------------
.. code-block:: bash

    pip install paramax


How it works
------------------
- :py:class:`~paramax.wrappers.AbstractUnwrappable` objects act as placeholders in the
  PyTree, defining the parameterizations.
- :py:func:`~paramax.wrappers.unwrap` applies the parameterizations, replacing the
  :py:class:`~paramax.wrappers.AbstractUnwrappable` objects.

A simple example of an :py:class:`~paramax.wrappers.AbstractUnwrappable`
is :py:class:`~paramax.wrappers.Parameterize`. This class takes a callable and any
positional or keyword arguments, which are stored and passed to the function when
unwrapping.


.. doctest::

   >>> import paramax
   >>> import jax.numpy as jnp
   >>> scale = jnp.ones(3)  # Keep this positive
   >>> constrained_scale = paramax.Parameterize(jnp.exp, jnp.log(scale))
   >>> model = ("abc", 1, constrained_scale)  # Any PyTree
   >>> paramax.unwrap(model)  # Unwraps any AbstractUnwrappables
   ('abc', 1, Array([1., 1., 1.], dtype=float32))


Many simple parameterizations can be handled with this class, for example,
we can parameterize a lower triangular matrix using

.. doctest::

   >>> import paramax
   >>> import jax.numpy as jnp
   >>> tril = jnp.tril(jnp.ones((3,3)))
   >>> tril = paramax.Parameterize(jnp.tril, tril)


See :doc:`/api/wrappers` for more :py:class:`~paramax.wrappers.AbstractUnwrappable`
objects.

When to unwrap
-------------------
- Unwrap whenever necessary, typically at the top of loss functions, functions or 
  methods requiring the parameterizations to have been applied.
- Unwrapping prior to a gradient computation used for optimization is usually a mistake!


.. toctree::
   :caption: API
   :maxdepth: 1
   :glob:

   api/wrappers
   api/utils

