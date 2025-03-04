import xlb
import trimesh
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import *
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warp.sim


# -------------------------- Simulation Setup --------------------------
wp.clear_kernel_cache()
# Grid parameters
grid_size_x, grid_size_y, grid_size_z = 512//2, 128//2, 128//2
grid_shape = (grid_size_x, grid_size_y, grid_size_z)

# Simulation Configuration
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32

velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
wind_speed = 0.02
num_steps = 100000
print_interval = 1000
post_process_interval = 1000

# Physical Parameters
Re = 50000.0
clength = grid_size_x - 1
visc = wind_speed * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

sim_dt = 3e-5

# Print simulation info
print("\n" + "=" * 50 + "\n")
print("Simulation Configuration:")
print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
print(f"Backend: {compute_backend}")
print(f"Velocity set: {velocity_set}")
print(f"Precision policy: {precision_policy}")
print(f"Prescribed velocity: {wind_speed}")
print(f"Reynolds number: {Re}")
print(f"Max iterations: {num_steps}")
print("\n" + "=" * 50 + "\n")

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create Grid
grid = grid_factory(grid_shape, compute_backend=compute_backend)

# Bounding box indices
box = grid.bounding_box_indices()
box_no_edge = grid.bounding_box_indices(remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()

# Create warp deformable shape
builder = wp.sim.ModelBuilder(up_vector=(0, 0, 1),gravity=0.0)
builder.add_soft_grid(
            pos=wp.vec3(grid_shape[0]/2, grid_shape[1]/2, grid_shape[2]/2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=grid_shape[0]//4*3,
            dim_y=grid_shape[1]//8,
            dim_z=grid_shape[1]//4*3,
            cell_x=1.0,
            cell_y=1.0,
            cell_z=1.0,
            density=1.0,
            k_mu=50000.0,
            k_lambda=20000.0,
            k_damp=0.0,
            fix_right=True,
        )
model = builder.finalize()
integrator = wp.sim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

def update_mesh(state):
    mesh_indices = jnp.arange(state.particle_q.shape[0])

    mesh = wp.Mesh(
        points=wp.array(state.particle_q, dtype=wp.vec3),
        indices=wp.array(mesh_indices, dtype=int),
        velocities=wp.zeros((mesh_indices.shape[0], 3), dtype=wp.vec3),
            )
    return mesh

mesh = update_mesh(state_0)

def update_fluid_sim(mesh):

    bc_left = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=inlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    bc_do_nothing = ExtrapolationOutflowBC(indices=outlet)
    bc_mesh = InterpolatedBounceBackMeshBC(mesh_vertices=state_0.particle_q,mesh_id=mesh.id)
    boundary_conditions = [bc_walls, bc_left, bc_do_nothing, bc_mesh]


    # Setup Stepper
    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    )

    # Prepare Fields
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    # Setup Momentum Transfer for Force Calculation
    bc_mesh = boundary_conditions[-1]
    momentum_transfer = MomentumTransfer(bc_mesh, compute_backend=ComputeBackend.JAX)

    return f_0, f_1, bc_mask, missing_mask, boundary_conditions, momentum_transfer, stepper

f_0, f_1, bc_mask, missing_mask,boundary_conditions,momentum_transfer, stepper = update_fluid_sim(mesh)

@wp.kernel
def interpolate_force(boundary_force, particle_q, force_interp):

    i,j,k = wp.tid()

    pos = particle_q[i,j,k] # position of particle
    # Closest Particle Indices
    i_min = wp.floor(pos[0])
    j_min = wp.floor(pos[1])
    k_min = wp.floor(pos[2])

    i_max = wp.ceil(pos[0])
    j_max = wp.ceil(pos[1])
    k_max = wp.ceil(pos[2])

    # Trilinear interpolation
    x_d = (pos[0]-wp.floor(pos[0]))
    y_d = (pos[1]-wp.floor(pos[1]))
    z_d = (pos[2]-wp.floor(pos[2]))

    # Interpolate the force
    c00=boundary_force[i_min,j_min,k_min]*(1-x_d)+boundary_force[i_max,j_min,k_min]*x_d
    c10=boundary_force[i_min,j_max,k_min]*(1-x_d)+boundary_force[i_max,j_max,k_min]*x_d
    c01=boundary_force[i_min,j_min,k_max]*(1-x_d)+boundary_force[i_max,j_min,k_max]*x_d
    c11=boundary_force[i_min,j_max,k_max]*(1-x_d)+boundary_force[i_max,j_max,k_max]*x_d

    c_0=c00*(1-y_d)+c10*y_d
    c_1=c01*(1-y_d)+c11*y_d

    force_interp[i,j,k] = c_0*(1-z_d)+c_1*z_d


def compute_force(
    f_0,
    f_1,
    momentum_transfer,
    missing_mask,
    bc_mask,
    state,
):

    boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)

    # Write an interpolation scheme to convert the force from grid-based to point-based
    force_interp = wp.zeros((state.particle_q.shape[0], 3), dtype=wp.vec3)

    wp.launch(
        kernel = interpolate_force,
        dim = state.particle_q.shape[0],
        inputs = [boundary_force, state.particle_q, force_interp],
        dim = 3,
    )

    return force_interp
    
def render(step,f_0, grid_shape, macro):
    # Compute macroscopic quantities
    if not isinstance(f_0, jnp.ndarray):
        f_0_jax = wp.to_jax(f_0)
    else:
        f_0_jax = f_0
    rho, u = macro(f_0_jax)
    # Remove boundary cells
    u = u[:, 1:-1, 1:-1, 1:-1]
    u_magnitude = jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

    fields = {"u_magnitude": u_magnitude,
              "rho":rho}

    # Save fields in VTK format
    save_fields_vtk(fields, timestep=step)
    # Save the u_magnitude slice at the mid y-plane
    mid_y = grid_shape[1] // 2
    save_image(fields["u_magnitude"][:, mid_y, :], timestep=step)



# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
)

# Initialize Lists to Store Coefficients and Time Steps
time_steps = []
drag_coefficients = []
lift_coefficients = []

# -------------------------- Simulation Loop --------------------------

start_time = time.time()
for step in range(num_steps):
    # Perform simulation step
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap the buffers

    # Compute the force
    force = compute_force(f_0, f_1, momentum_transfer, missing_mask, bc_mask, state_0)

    # Integrate Solid 
    wp.sim.collide(model,state_0)
    state_0.clear_forces()
    state_1.clear_forces()

    integrator.simulate(model, state_0, state_1,sim_dt)

    # Swap States
    state_0, state_1 = state_1, state_0

    # Update Mesh
    mesh = update_mesh(state_0)

    # Update Fluid Simulation
    f_0, f_1, bc_mask, missing_mask, boundary_conditions, momentum_transfer, stepper = update_fluid_sim(mesh)

    # Print progress at intervals
    if step % print_interval == 0:
        #if compute_backend == ComputeBackend.WARP:
            #wp.synchronize()
        elapsed_time = time.time() - start_time
        print(f"Iteration: {step}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")
        start_time = time.time()

    #Post-process at intervals and final step
    if (step % post_process_interval == 0) or (step == num_steps - 1):
        render(
            step,
            f_0,
            grid_shape,
            macro
        )


print("Simulation completed successfully.")