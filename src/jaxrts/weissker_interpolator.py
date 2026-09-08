"""
Interpolator, according to :cite:`Weissker.2009`, currently targeted at S_ii.

Since S_ii does not obey a sum-rule, the save the integral for each grid point
and interpolate to the overall signal strength.
"""

import functools
import jax
import jax.numpy as jnp
from quadax import cumulative_trapezoid
from .units import Quantity, ureg


@jax.tree_util.register_pytree_node_class
class SiiInterpolator:
    def __init__(self, Sii: Quantity, k: Quantity, *free_variables):
        """
        Sii shape: [i, j, k, variables]
        """
        self.k = k.m_as(1 / ureg.angstrom)
        self.variables_units = [v.units for v in free_variables]
        self.variables = [
            v.m_as(u)
            for (v, u) in zip(
                free_variables, self.variables_units, strict=True
            )
        ]
        integral = cumulative_trapezoid(
            Sii.m_as(ureg.dimensionless),
            x=self.k,
            axis=2,
            initial=0.0,
        )
        norm = integral[:, :, -1, :]
        self.interpolator = VRegularGridInterpolator(
            self.variables, integral / norm[:, :, jnp.newaxis, :]
        )
        self.norm_interpolator = VRegularGridInterpolator(self.variables, norm)

    @jax.jit
    def __call__(self, point):
        _point = jnp.array(
            [
                p.m_as(u)
                for (p, u) in zip(point, self.variables_units, strict=True)
            ]
        )
        interpolation = self.interpolator(_point)
        norm = self.norm_interpolator(_point)
        Sii = jnp.gradient(interpolation, self.k, axis=2)
        return (
            Sii
            * (norm / interpolation[:, :, -1])[:, :, jnp.newaxis]
            * 1
            * ureg.dimensionless
        )

    def tree_flatten(self):
        return (
            self.k,
            self.variables,
            self.interpolator,
            self.norm_interpolator,
        ), (self.variables_units,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (
            obj.k,
            obj.variables,
            obj.interpolator,
            obj.norm_interpolator,
        ) = children
        (obj.variables_units,) = aux_data
        return obj


@jax.tree_util.register_pytree_node_class
class VRegularGridInterpolator:
    """
    Vectorized RegularGridInterpolator, with a linear interpolation:
    Interpolated linarly over the last indices of `values`, i.e.,
    `values.shape` should be `[*vectorized_dims.shape, *points.shape]`, and the
    output of a call will be of shape vectorized_dims.shape.
    """

    def __init__(self, points, values):
        self.points = points
        self.values = values

    @jax.jit
    def __call__(self, point):
        ndim = len(self.points)
        lead = self.values.ndim - ndim  # number of vectorized channel axes

        # type casting is required to ensure cube obly gets the same type
        lo, frac = [], []
        for p, x in zip(self.points, point):
            i = jnp.clip(jnp.searchsorted(p, x) - 1, 0, p.shape[0] - 2)
            lo.append(i.astype(jnp.int32))
            frac.append((x - p[i]) / (p[i + 1] - p[i]))

        weights = functools.reduce(
            lambda a, b: jnp.tensordot(a, b, axes=0),
            (jnp.stack([1 - t, t]) for t in frac),
        )

        zeros = jnp.zeros((lead,), dtype=jnp.int32)
        start = tuple(zeros) + tuple(lo)
        size = self.values.shape[:lead] + (2,) * ndim
        cube = jax.lax.dynamic_slice(self.values, start, size)

        return jnp.tensordot(cube, weights, axes=ndim)

    def tree_flatten(self):
        return (self.points, self.values), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.points, obj.values = children
        return obj
