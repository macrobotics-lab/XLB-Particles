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
        u_wall=None,
        sphere_c=None,
        sphere_r=None,
    ):
        self.u = u_wall
        self.sphere_c = sphere_c
        self.sphere_r = sphere_r
        self.mesh_vertices = mesh_vertices
        
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
        _w = self.velocity_set.w

        #Weight function for a mesh using raytracing

        @wp.func
        def calculate_weight(
            mesh_id: wp.uint64,
            origin: Any,
            direction: Any,
            i: Any,
        ):
            # Compute the weight associated with any given direction
            query = wp.mesh_query_ray(mesh_id=mesh_id,start=origin,dir=direction[:,i],max_t = 1.5)
            if query.result:
                weight = query.t
                face = query.face
                u = query.u
                v = query.v
                return weight,face,u,v
            else :
                raise ValueError("No intersection found with Mesh for the given direction")

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
            mesh_indices = range(self.mesh_vertices.shape[0]) # May have to be a wp array to get the shape inside a wp function 
            mesh = wp.Mesh(
            points=wp.array(self.mesh_vertices, dtype=wp.vec3),
            indices=wp.array(mesh_indices, dtype=int),
            )
            # Post-streaming values are only modified at missing direction
            _f = f_post
            for l in range(self.velocity_set.q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    # Get the pre-streaming distribution function in oppisite direction
                    weight,face,u,v = calculate_weight(mesh.id,index,_c, _opp_indices[l])
                    wall_velocity = wp.mesh_eval_velocity(id=mesh.id,face=face,bary_u=u,bary_v=v)
                    _f[l] = 2.0*weight/(1.0+2.0*weight)*f_post[l] + 1.0/(1.0+2.0*weight)*f_pre[_opp_indices[l]] + 6.0*weight/(1.0-2.0*weight)*_w[l]*wp.dot(wall_velocity,_c[:,l])

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
