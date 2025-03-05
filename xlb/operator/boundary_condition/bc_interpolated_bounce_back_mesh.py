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


class InterpolatedBounceBackMeshBC(BoundaryCondition):
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
        mesh_id=None,
    ):
        self.mesh_vertices=mesh_vertices
        self.mesh_id = wp.uint64(mesh_id)


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
        boundary = lax.broadcast_in_dim(
            boundary, new_shape, tuple(range(self.velocity_set.d + 1))
        )
        return jnp.where(
            jnp.logical_and(missing_mask, boundary),
            f_pre[self.velocity_set.opp_indices],
            f_post,
        )

    def _construct_warp(self):
        # Set local constants
        _opp_indices = self.velocity_set.opp_indices
        _c_float = self.velocity_set.c_float
        _w = self.velocity_set.w
        mesh_id = self.mesh_id

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
            for l in range(self.velocity_set.q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    # Get the pre-streaming distribution function in opposite direction
                    dir = wp.vec3f(
                        _c_float[0, _opp_indices[l]],
                        _c_float[1, _opp_indices[l]],
                        _c_float[2, _opp_indices[l]],
                    )
                    start = wp.vec3f(
                        wp.float32(index[0]), wp.float32(index[1]), wp.float32(index[2])
                    )
                    length = wp.length(dir)

                    t = float(0.0)      # hit distance along ray
                    u = float(0.0)      # hit face barycentric u
                    v = float(0.0)      # hit face barycentric v
                    sign = float(0.0)   # hit face sign
                    n = wp.vec3()       # hit face normal
                    f = int(0)          # hit face index
                    
                    wp.mesh_query_ray(mesh_id, start, dir, length,t,u,v,sign,n,f)
                    f_post[l] = (
                                2.0 * t/ (1.0 + 2.0 * t) * f_post[l]
                                + 1.0 / (1.0 + 2.0 * t) * f_pre[_opp_indices[l]]
                                + 6.0* t/ (1.0 - 2.0 * t)* _w[l]* wp.dot(wp.mesh_eval_velocity(mesh_id, f, u, v), dir)
                            )

            return f_post

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
