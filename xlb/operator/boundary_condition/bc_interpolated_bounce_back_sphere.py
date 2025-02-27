"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)


class InterpolatedBounceBackSphereBC(BoundaryCondition):
    """
    Halfway Bounce-back boundary condition for a lattice Boltzmann method simulation.

    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,        
        u_wall=None,
        sphere_c=None,
        sphere_r=None,
    ):
        self.u = u_wall
        self.sphere_c = sphere_c
        self.sphere_r = sphere_r
        
        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

        # This BC needs padding for finding missing directions when imposed on a geometry that is in the domain interior
        self.needs_padding = True

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
        return jnp.where(
            jnp.logical_and(missing_mask, boundary),
            f_pre[self.velocity_set.opp_indices],
            f_post,
        )

    def _construct_warp(self):
        # Set local constants
        _opp_indices = self.velocity_set.opp_indices
        _c = self.velocity_set.c
        _sphere_c = (wp.vec3i(self.sphere_c[0], self.sphere_c[1], self.sphere_c[2]))
        _sphere_r = self.compute_dtype(self.sphere_r)
        _w = self.velocity_set.w

        #Weight function for a sphere

        @wp.func
        def calculate_weight(
            u: Any,
            o: Any,
            c: Any,
            r: Any,
            l: Any,
        ):
            # Compute the weight associated with any given direction

            un = wp.vec3(
                self.compute_dtype(u[0, l]),
                self.compute_dtype(u[1, l]),
                self.compute_dtype(u[2, l]),
            )
            on = wp.vec3(
                self.compute_dtype(o[0]),
                self.compute_dtype(o[1]),
                self.compute_dtype(o[2]),
            )
            cn = wp.vec3(
                self.compute_dtype(c[0]),
                self.compute_dtype(c[1]),
                self.compute_dtype(c[2]),
            )

            u_hat = un / wp.length(un)
            a = wp.dot(u_hat, (on - cn))
            delta = a**2.0 - (wp.length(on - cn) ** 2.0 - r**2.0)

            d = wp.min(-a + wp.sqrt(delta), -a - wp.sqrt(delta))
            return  d / wp.length(un)

        # Construct the functional for this BC
        @wp.func
        def functional(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post
            for l in range(self.velocity_set.q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    # Get the pre-streaming distribution function in oppisite direction
                    weight = calculate_weight(_c,index,_sphere_c,_sphere_r,_opp_indices[l])
                    _f[l] = 2.0*weight/(1.0+2.0*weight)*f_post[l] + 1.0/(1.0+2.0*weight)*f_pre[_opp_indices[l]]+6.0*weight/(1.0-2.0*weight)*_w[l]*wp.dot(self.u,_c[:,l])

            return _f

        kernel = self._construct_kernel(functional)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post
