"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any
from collections import Counter
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.macroscopic.zero_moment import ZeroMoment
from xlb.operator.macroscopic.first_moment import FirstMoment
from xlb.operator.macroscopic.second_moment import SecondMoment as MomentumFlux
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)


class TammMothSmithBC(BoundaryCondition):
    """
    Purpose: Modificaiton of existing bc_grads_approximation.py to make it more in line with the local Tamm Moth Smith Boundary Condition
    [1] S. Chikatamarla and I. Karlin, "Entropic lattice Boltzmann Methods for turbulent flow simulations:
        Boundary Conditions", Physica A. 392 (2013).

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
        # TODO: the input velocity must be suitably stored elesewhere when mesh is moving.

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

        # Instantiate the operator for computing macroscopic values
        self.macroscopic = Macroscopic()
        self.zero_moment = ZeroMoment()
        self.first_moment = FirstMoment()
        self.equilibrium = QuadraticEquilibrium()
        self.momentum_flux = MomentumFlux()

        # This BC needs implicit distance to the mesh
        self.needs_mesh_distance = True

        # If this BC is defined using indices, it would need padding in order to find missing directions
        # when imposed on a geometry that is in the domain interior
        if self.mesh_vertices is None:
            assert self.indices is not None
            self.needs_padding = True

        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This BC is not implemented in 2D!")

        # if indices is not None:
        #     # this BC would be limited to stationary boundaries
        #     # assert mesh_vertices is None
        # if mesh_vertices is not None:
        #     # this BC would be applicable for stationary and moving boundaries
        #     assert indices is None
        #     if mesh_velocity_function is not None:
        #         # mesh is moving and/or deforming

        assert (
            self.compute_backend == ComputeBackend.WARP
        ), "This BC is currently only implemented with the Warp backend!"

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # TODO
        raise NotImplementedError(
            f"Operation {self.__class__.__name} not implemented in JAX!"
        )
        return

    def _construct_warp(self):
        # Set local variables and constants
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _d = self.velocity_set.d
        _w = self.velocity_set.w
        _qi = self.velocity_set.qi
        _opp_indices = self.velocity_set.opp_indices
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _u_wall = (
            _u_vec(self.u[0], self.u[1], self.u[2])
            if _d == 3
            else _u_vec(self.u[0], self.u[1])
        )
        _sphere_c = (
            wp.vec3i(self.sphere_c[0], self.sphere_c[1], self.sphere_c[2])
            if _d == 3
            else wp.vec2i(self.sphere_c[0], self.sphere_c[1])
        )
        _sphere_r = self.compute_dtype(self.sphere_r)
        # diagonal = wp.vec3i(0, 3, 5) if _d == 3 else wp.vec2i(0, 2)

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

        # Construct the functionals for this BC
        @wp.func
        def functional_method(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Strategy:
            # 1) "weights" are computed based on the center position of an input sphere
            # 2) Given "weights", "u_w" (input to the BC) and "u_f" (computed from f_aux), compute "u_target" as per Eq (14)
            #    NOTE: in the original paper "u_target" is associated with the previous time step not current time.
            # 3) Compute rho_bb from interpolated bb
            # 4) Compute rho_s from u_w
            # 5) Compute feq using feq = self.equilibrium(rho_target, u_target) for missing pops
            # 6) Apply TMS for all pops

            _f_nbr = _f_vec()
            u_target = _u_vec(0.0, 0.0, 0.0)  # if _d == 3 else _u_vec(0.0, 0.0)
            num_missing = self.compute_dtype(0)
            one = self.compute_dtype(1.0)
            rho_target = self.compute_dtype(0)
            for l in range(_q):
                # If the mask is missing then take the target population 
                if missing_mask[l] == wp.uint8(1):
                    #Density from Bounce back (non interpolated for now)
                    rho_target += f_pre[_opp_indices[l]]
                else:
                    rho_target += f_pre[l]

            #Velocity and population update loop
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    # Find the neighbour and its velocity value
                    for ll in range(_q):
                        # f_0 is the post-collision values of the current time-step
                        # Get index associated with the fluid neighbours
                        fluid_nbr_index = type(index)()
                        for d in range(_d):
                            fluid_nbr_index[d] = index[d] + _c[d, l]
                        # The following is the post-collision values of the fluid neighbor cell
                        _f_nbr[ll] = self.compute_dtype(
                            f_0[
                                ll,
                                fluid_nbr_index[0],
                                fluid_nbr_index[1],
                                fluid_nbr_index[2],
                            ]
                        )

                    # Compute the velocity vector at the fluid neighbouring cells
                    _, u_f = self.macroscopic.warp_functional(_f_nbr)

                    # Record the number of missing directions
                    num_missing += 1.0

                    # Interpolation weights computed for a sphere.
                    weight = calculate_weight(
                        _c, index, _sphere_c, _sphere_r,_opp_indices[l]
                    )
                    # Given "weights", "u_w" (input to the BC) and "u_f" S(computed from f_aux), compute "u_target" as per Eq (14)
                    for d in range(_d):
                        u_target[d] += (weight * u_f[d] + _u_wall[d]) / (one + weight)
                    

                    # Add contribution due to moving_wall to f_missing
                    cu = self.compute_dtype(0.0)
                    for d in range(_d):
                        cu += self.compute_dtype(6.0) * _w[l]*_u_wall[d] * self.compute_dtype(_c[d, l])

            for d in range(_d):
                u_target[d] /= num_missing
            
            rho_local, u_local = self.macroscopic.warp_functional(f_post)
            f_post = self.equilibrium.warp_functional(rho_target, u_target) - self.equilibrium.warp_functional(rho_local, u_local)

            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    f_post[l] = f_post[l] + self.equilibrium.warp_functional(rho_target, u_target)[l]

            return f_post

        functional = functional_method

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
