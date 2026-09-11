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
    def __init__(
        self,
        Sii: Quantity,
        k: Quantity,
        *free_variables,
        grid_length: int = 1000,
    ):
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
        integral /= norm[:, :, jnp.newaxis, :]

        flipped_integral = _invert_k_axis(
            integral, jnp.linspace(0, 1, grid_length), self.k
        )

        self.interpolator = VRegularGridLinearInterpolator(
            self.variables, flipped_integral
        )
        self.norm_interpolator = VRegularGridLinearInterpolator(
            self.variables, norm
        )

    @jax.jit
    def __call__(self, point):
        _point = jnp.array(
            [
                p.m_as(u)
                for (p, u) in zip(point, self.variables_units, strict=True)
            ]
        )
        flipped_interpolation = self.interpolator(_point)
        interpolation = _invert_k_axis(
            flipped_interpolation,
            self.k,
            jnp.linspace(0, 1, flipped_interpolation.shape[2]),
        )

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
class VRegularGridLinearInterpolator:
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

        # type casting is required to ensure cube only gets the same type
        lo, w_vecs, sizes = [], [], []
        for p, x in zip(self.points, point):
            n = p.shape[0]  # static (Python int), safe to branch on
            # If there is only one grid point per dimension, return only this
            # function.
            if n == 1:
                i = jnp.zeros((), dtype=jnp.int32)
                w = jnp.ones((1,), dtype=x.dtype)
                sizes.append(1)
            else:
                i = jnp.clip(jnp.searchsorted(p, x) - 1, 0, n - 2).astype(
                    jnp.int32
                )
                t = (x - p[i]) / (p[i + 1] - p[i])
                w = jnp.stack([1 - t, t])
                sizes.append(2)
            lo.append(i)
            w_vecs.append(w)

        weights = functools.reduce(
            lambda a, b: jnp.tensordot(a, b, axes=0), w_vecs
        )

        zeros = jnp.zeros((lead,), dtype=jnp.int32)
        start = tuple(zeros) + tuple(lo)
        size = self.values.shape[:lead] + tuple(sizes)
        cube = jax.lax.dynamic_slice(self.values, start, size)

        return jnp.tensordot(cube, weights, axes=ndim)

    def tree_flatten(self):
        return (self.points, self.values), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.points, obj.values = children
        return obj


def _invert_k_axis(Sii, grid, x):
    """
    Invert the k axis (axis 2) of Sii by interpolating to the values on grid
    """
    # Move the axis to invert to the end
    y = jnp.moveaxis(Sii, 2, -1)
    shape = y.shape[:-1]

    y = y.reshape(-1, y.shape[-1])

    def inverse_interp(y_curve):
        idx = jnp.searchsorted(y_curve, grid, side="left")
        idx = jnp.clip(idx, 1, y_curve.size - 1)

        y0 = y_curve[idx - 1]
        y1 = y_curve[idx]

        x0 = x[idx - 1]
        x1 = x[idx]

        return x0 + (grid - y0) * (x1 - x0) / (y1 - y0)

    result = jax.vmap(inverse_interp)(y)

    # Restore original order
    result = result.reshape(*shape, grid.size)
    return jnp.moveaxis(result, -1, 2)
