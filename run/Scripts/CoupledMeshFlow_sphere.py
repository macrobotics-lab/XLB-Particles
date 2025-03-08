import xlb
import meshio
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
import warp.sim.render
from xlb.operator.boundary_masker import MeshBoundaryMasker


# -------------------------- Simulation Setup --------------------------
wp.clear_kernel_cache()
# wp.set_mempool_release_threshold("cuda:0", 0)
# wp.config.verbose = True
# wp.config.print_launches = True
# wp.config.mode = "debug"
# wp.config.verify_fp = True
# wp.config.verify_cuda = True
# Grid parameters
grid_size_x, grid_size_y, grid_size_z = 512//2, 64, 64
grid_shape = (grid_size_x, grid_size_y, grid_size_z)

# Simulation Configuration
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32

velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend)
wind_speed = 0.02
num_steps = 100000
print_interval = 1000
post_process_interval = 1000

# Physical Parameters
Re = 8
clength = 8
visc = 0.1
omega = 1.0 / (3.0 * visc + 0.5)

sim_dt = 2e-5

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
mesh = meshio.read('../Inputs/Sphere_refined.msh')
mesh_points = mesh.points
mesh_indices = np.array(mesh.cells_dict['tetra'],dtype=int).ravel()
mesh_triangles = np.array(mesh.cells_dict['triangle'],dtype=int)
mesh_points -= mesh_points.min(axis=0)
mesh_points = mesh_points*15.
mesh_center = mesh_points.max(axis=0)
shift = np.array([grid_shape[0] // 5, (grid_shape[1] - mesh_center[1]) // 2, (grid_shape[2] - mesh_center[2])//2])

builder = warp.sim.ModelBuilder(up_vector=(1., 0, 0),gravity=0.)
builder.add_soft_mesh(
            vertices = mesh_points,
            indices = mesh_indices,
            pos=shift,
            rot=wp.quat_identity(),
            vel = wp.vec3(0.0,0.0,0.0),
            scale=1.,                
            k_mu=50000.,
            k_lambda=20000.,
            k_damp=0.,
            density=1.1,
        )
#builder.add_triangles(mesh_triangles[:,0],mesh_triangles[:,1],mesh_triangles[:,2]) ## Adding surface triangles to soft mesh

model = builder.finalize()
integrator = warp.sim.SemiImplicitIntegrator()

state_0 = model.state()
state_1 = model.state()

print(f"Initialized {state_0.body_count} Bodies")
print(f"Initialized {state_0.particle_count} Particles")
print(f"Initialized {model.tet_count} Tetrahedra")
print(f"Initialized {model.tri_count} Triangles")

#Deconstruct tet mesh into triangles 
mesh_indices = np.array(mesh.cells_dict['tetra'],dtype=int)
inner_indices = np.zeros((4*mesh_indices.shape[0],3),dtype=int)

for i in range(4*mesh_indices.shape[0]): 
    for j in range(4):
        inner_indices[i] = np.roll(mesh_indices[i//4,:],j)[0:3]

indices = wp.array(np.append(mesh_triangles,inner_indices,axis=0),dtype=int).flatten()


def update_mesh(mesh, model,state):
    if mesh == None:
        
        mesh = wp.Mesh(
            points=state.particle_q,
            indices= indices,
            velocities=state.particle_qd,
                )
        return mesh
    else:
        mesh.points = state.particle_q
        mesh.velocities = state.particle_qd
        mesh.refit()
        return mesh

mesh = update_mesh(None, model,state_0)

bc_left = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=inlet)
#bc_left = ExtrapolationOutflowBC(indices=inlet)
bc_walls = HalfwayBounceBackBC(indices=walls)
bc_do_nothing = ExtrapolationOutflowBC(indices=outlet)
#bc_mesh = InterpolatedBounceBackMeshBC(mesh_vertices=1,mesh_id=mesh.id)
bc_mesh = HalfwayBounceBackBC(mesh_vertices=1,mesh_id=mesh.id)
boundary_conditions_static = [bc_walls, bc_left, bc_do_nothing]
boundary_conditions = [bc_walls, bc_left, bc_do_nothing, bc_mesh]


stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions_static,
        collision_type="BGK",
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
def interpolate_force(boundary_force : wp.array(dtype=wp.vec3,ndim=3), particle_q : wp.array(dtype=wp.vec3), query_indices: wp.array(dtype=int), force_interp: wp.array(dtype=wp.vec3)):

    i= wp.tid()

    pos = particle_q[query_indices[i]] # position of particle
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

    force_interp[query_indices[i]] = c_0*(1.-z_d)+c_1*z_d


def compute_force(
    f_0,
    f_1,
    momentum_transfer,
    missing_mask,
    bc_mask,
    state,
    query_indices,
    force,
    force_interp,
):
    force.zero_()
    force = momentum_transfer(f_0, f_1, bc_mask, missing_mask,force)
    # Write an interpolation scheme to convert the force from grid-based to point-based
    wp.launch(
        kernel = interpolate_force,
        dim = query_indices.shape[0],
        inputs = [force, state.particle_q, query_indices, force_interp],
    )

    return force, force_interp
    
def render(renderer, step,f_0, grid_shape, macro,rho,u,force,force_interp,state_0,state_1,):
    # Compute macroscopic quantities
    _, u = macro(f_0,rho,u)
    # Remove boundary cells
    u = u.numpy()
    force = force.numpy()
    force_interp = force_interp.numpy()

    u = u[:, 1:-1, 1:-1, 1:-1]
    force = np.moveaxis(force,-1,0)
    force = force[:,1:-1, 1:-1, 1:-1]

    print(f"Net Force from MEM : {np.sum(force,axis=(1,2,3))}")
    print(f"Net Force from Interpolation : {np.sum(state_0.particle_f.numpy(),axis=0)}")
    print(f"Mean Velocity of Mesh : {np.mean(force_interp,axis=0)}")
    print(f"Change in Mean Particle Position: {np.mean(state_0.particle_q.numpy(),axis=0)-np.mean(state_1.particle_q.numpy(),axis=0)}")

    u_magnitude = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    f_magnitude = np.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2)


    fields = {"force": f_magnitude, "u_magnitude": u_magnitude}

    # Save fields in VTK format
    save_fields_vtk(fields, timestep=step)
    # Save the u_magnitude slice at the mid y-plane
    mid_y = grid_shape[1] // 2
    save_image(fields["u_magnitude"][:, mid_y, :], timestep=step)


    # renderer.begin_frame(step)
    # renderer.render(state_0)
    # renderer.end_frame()

# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.WARP,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP),
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
renderer = None #warp.sim.render.SimRendererOpenGL(model,'./', )
query_indices = wp.array(mesh_triangles.flatten(),dtype=int)
# -------------------------- Simulation Loop --------------------------
dx = 0

start_time = time.time()
for step in range(num_steps):
    # Perform simulation step
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap the buffers

    # Compute the force
    state_0.clear_forces()
    force.zero_()
    force, force_interp = compute_force(f_0, f_1, momentum_transfer, missing_mask, bc_mask, state_0,query_indices,force,force_interp)
    state_0.particle_f = force_interp

    # Integrate Solid 

    #integrator.simulate(model, state_0, state_1,sim_dt)

    # Swap States
    #state_0, state_1 = state_1, state_0

    #mesh = update_mesh(mesh,model,state_0)
    dx=0
    
        # Update Fluid Simulation
    #bc_mesh.mesh_vertices =1
    #bc_mesh.mesh_id = mesh.id

        # Update Fields
    #bc_mask, missing_mask = update_dynamic_BC(mesh_masker, bc_mesh,bc_mask_static,missing_mask_static)

        # Update Momentum Transfer for Force Calculation
    #momentum_transfer.no_slip_bc_instance = bc_mesh

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
            renderer,
            step,
            f_0,
            grid_shape,
            macro,
            rho,
            u,
            force,
            state_0.particle_qd,
            state_0,
            state_1,
        )


print("Simulation completed successfully.")