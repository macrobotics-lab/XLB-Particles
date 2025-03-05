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
from xlb.operator.boundary_masker import MeshBoundaryMasker


# -------------------------- Simulation Setup --------------------------
wp.clear_kernel_cache()
# wp.set_mempool_release_threshold("cuda:0", 0)
# wp.config.verbose = True
# wp.config.print_launches = True
# wp.config.mode = "debug"
# wp.config.verify_cuda = True
# Grid parameters
grid_size_x, grid_size_y, grid_size_z = 512//2, 128//2, 128
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
Re = 5000.0
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
builder = wp.sim.ModelBuilder(up_vector=(0, 0, 1.),gravity=0.0)
builder.add_soft_grid(
            pos=wp.vec3(grid_shape[0]//8, grid_shape[1]//4, grid_shape[2]//2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(grid_shape[0]//4),
            dim_y=int(grid_shape[1]//2),
            dim_z=int(grid_shape[2]//32),
            cell_x=int(1),
            cell_y=int(1),
            cell_z=int(1),
            density=1.0,
            k_mu=1.0,
            k_lambda=1.0,
            k_damp=0.0,
            fix_left=True,
        )
model = builder.finalize()
integrator = wp.sim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

def update_mesh(mesh, model,state):
    if mesh == None:
        mesh = wp.Mesh(
            points=state.particle_q,
            indices=model.tri_indices.flatten(),
            velocities=state.particle_qd,
                )
        return mesh
    else:
        mesh.points = state.particle_q
        mesh.refit()
    return mesh

mesh = update_mesh(None, model,state_0)

bc_left = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)
bc_do_nothing = ExtrapolationOutflowBC(indices=outlet)
bc_mesh = InterpolatedBounceBackMeshBC(mesh_vertices=1,mesh_id=mesh.id)
boundary_conditions_static = [bc_walls, bc_left, bc_do_nothing]
boundary_conditions = [bc_walls, bc_left, bc_do_nothing, bc_mesh]


stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions_static,
        collision_type="KBC",
    )
f_0, f_1, bc_mask_static, missing_mask_static = stepper.prepare_fields()

bc_walls.indices = walls
bc_left.indices = inlet
bc_do_nothing.indices = outlet

stepper.boundary_conditions = boundary_conditions

mesh_masker =  MeshBoundaryMasker(
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend,)

def update_dynamic_BC(mesh_masker, bc_mesh,bc_mask_static,missing_mask_static):

    bc_mask, missing_mask = mesh_masker(bc_mesh, bc_mask_static, missing_mask_static)
    return bc_mask, missing_mask

bc_mask, missing_mask = update_dynamic_BC(mesh_masker, bc_mesh,bc_mask_static,missing_mask_static)

@wp.kernel
def interpolate_force(boundary_force : wp.array(dtype=wp.vec3,ndim=3), particle_q : wp.array(dtype=wp.vec3), force_interp: wp.array(dtype=wp.vec3)):

    i= wp.tid()

    pos = particle_q[i] # position of particle
    # Closest Particle Indices
    i_min = int(wp.floor(pos[0]))
    j_min = int(wp.floor(pos[1]))
    k_min = int(wp.floor(pos[2]))

    i_max = int(wp.ceil(pos[0]))
    j_max = int(wp.ceil(pos[1]))
    k_max = int(wp.ceil(pos[2]))

    # Trilinear interpolation
    x_d = (pos[0]-wp.floor(pos[0]))
    y_d = (pos[1]-wp.floor(pos[1]))
    z_d = (pos[2]-wp.floor(pos[2]))

    # Interpolate the force
    c00=boundary_force[i_min,j_min,k_min]*(1.-x_d)+boundary_force[i_max,j_min,k_min]*x_d
    c10=boundary_force[i_min,j_max,k_min]*(1.-x_d)+boundary_force[i_max,j_max,k_min]*x_d
    c01=boundary_force[i_min,j_min,k_max]*(1.-x_d)+boundary_force[i_max,j_min,k_max]*x_d
    c11=boundary_force[i_min,j_max,k_max]*(1.-x_d)+boundary_force[i_max,j_max,k_max]*x_d

    c_0=c00*(1.-y_d)+c10*y_d
    c_1=c01*(1.-y_d)+c11*y_d

    force_interp[i] = c_0*(1.-z_d)+c_1*z_d


def compute_force(
    f_0,
    f_1,
    momentum_transfer,
    missing_mask,
    bc_mask,
    state,
    force,
    force_interp,
):
    force = momentum_transfer(f_0, f_1, bc_mask, missing_mask,force)
    # Write an interpolation scheme to convert the force from grid-based to point-based
    wp.launch(
        kernel = interpolate_force,
        dim = state.particle_q.shape[0],
        inputs = [force, state.particle_q, force_interp],
    )

    return force, force_interp
    
def render(step,f_0, grid_shape, macro,rho,u,force,force_interp):
    # Compute macroscopic quantities
    _, u = macro(f_0,rho,u)
    # Remove boundary cells
    u = u.numpy()
    force = force.numpy()
    force_interp = force_interp.numpy()

    u = u[:, 1:-1, 1:-1, 1:-1]
    force = np.moveaxis(force,-1,0)
    force = force[:,1:-1, 1:-1, 1:-1]
    

    print(f"Maximum Force from MEM : {np.max(force)}")
    print(f"Mean Velocity of Mesh : {np.mean(force_interp)}")

    u_magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    f_magnitude = np.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2)


    fields = {"force": f_magnitude, "u_magnitude": u_magnitude}

    # Save fields in VTK format
    save_fields_vtk(fields, timestep=step)
    # Save the u_magnitude slice at the mid y-plane
    mid_y = grid_shape[1] // 2
    save_image(fields["u_magnitude"][:, mid_y, :], timestep=step)



# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.WARP,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP),
)

# Initialize Lists to Store Coefficients and Time Steps
time_steps = []
drag_coefficients = []
lift_coefficients = []

rho = wp.zeros((1,grid_shape[0],grid_shape[1],grid_shape[2]), dtype=wp.float32)
u = wp.zeros((3,grid_shape[0],grid_shape[1],grid_shape[2]),dtype=wp.float32)
force_interp = wp.zeros(state_0.particle_q.shape[0], dtype=wp.vec3)
force = wp.zeros((grid_shape[0],grid_shape[1],grid_shape[2]),dtype=wp.vec3)
momentum_transfer = MomentumTransfer(boundary_conditions[-1], compute_backend=ComputeBackend.WARP,force=force)
# -------------------------- Simulation Loop --------------------------

start_time = time.time()
for step in range(num_steps):
    # Perform simulation step
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap the buffers

    # Compute the force
    force_interp.zero_()
    force, force_interp = compute_force(f_0, f_1, momentum_transfer, missing_mask, bc_mask, state_0,force,force_interp)
    state_0.particle_f = force_interp

    # # Integrate Solid 
    integrator.simulate(model, state_0, state_1,sim_dt)

    # Swap States
    state_0, state_1 = state_1, state_0

    # Update Mesh
    
    mesh = update_mesh(mesh,model,state_0)
 
    # Update Fluid Simulation
    bc_mesh.mesh_vertices =1
    bc_mesh.mesh_id = mesh.id

    # Update Fields
    bc_mask, missing_mask = update_dynamic_BC(mesh_masker, bc_mesh,bc_mask_static,missing_mask_static)

    # Update Momentum Transfer for Force Calculation
    momentum_transfer.no_slip_bc_instance = bc_mesh

    # Print progress at intervals
    if step % print_interval == 0:
        if compute_backend == ComputeBackend.WARP:
            wp.synchronize()
        elapsed_time = time.time() - start_time
        print(f"Iteration: {step}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")
        start_time = time.time()

    #Post-process at intervals and final step
    if (step % post_process_interval == 0) or (step == num_steps - 1):
        render(
            step,
            f_0,
            grid_shape,
            macro,
            rho,
            u,
            force,
            state_0.particle_qd
        )


print("Simulation completed successfully.")