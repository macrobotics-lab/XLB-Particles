import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    TammMothSmithBC,
    InterpolatedBounceBackSphereBC,
    GradsApproximationBC,
)
from xlb.operator.force import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

# -------------------------- Simulation Setup --------------------------
grid_shape = (512 // 2, 128 // 2, 128 // 2)
Re = 100.0
u_max = 0.04
sphere_radius = grid_shape[1] // 12
visc = u_max * sphere_radius / Re
omega = 1.0 / (3.0 * visc + 0.5)
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D3Q19(
    precision_policy=precision_policy, compute_backend=compute_backend
)

num_steps = 20000
post_process_interval = 1000

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create Grid
grid = grid_factory(grid_shape, compute_backend=compute_backend)

# Define Boundary Indices
box = grid.bounding_box_indices()
box_no_edge = grid.bounding_box_indices(remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [
    box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i]
    for i in range(velocity_set.d)
]
walls = np.unique(np.array(walls), axis=-1).tolist()

sphere_radius = grid_shape[1] // 12
x = np.arange(grid_shape[0])
y = np.arange(grid_shape[1])
z = np.arange(grid_shape[2])
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
sphere_center = (grid_shape[0] // 6, grid_shape[1] // 2, grid_shape[2] // 2)
indices = np.where(
    (X - grid_shape[0] // 6) ** 2
    + (Y - grid_shape[1] // 2) ** 2
    + (Z - grid_shape[2] // 2) ** 2
    < sphere_radius**2
)
sphere = [tuple(indices[i]) for i in range(velocity_set.d)]


# Define Boundary Conditions
def bc_profile():
    H_y = float(grid_shape[1] - 1)  # Height in y direction
    H_z = float(grid_shape[2] - 1)  # Height in z direction

    if compute_backend == ComputeBackend.JAX:

        def bc_profile_jax():
            y = jnp.arange(grid_shape[1])
            z = jnp.arange(grid_shape[2])
            Y, Z = jnp.meshgrid(y, z, indexing="ij")

            # Calculate normalized distance from center
            y_center = Y - (H_y / 2.0)
            z_center = Z - (H_z / 2.0)
            r_squared = (2.0 * y_center / H_y) ** 2.0 + (2.0 * z_center / H_z) ** 2.0

            # Parabolic profile for x velocity, zero for y and z
            u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
            u_y = jnp.zeros_like(u_x)
            u_z = jnp.zeros_like(u_x)

            return jnp.stack([u_x, u_y, u_z])

        return bc_profile_jax

    elif compute_backend == ComputeBackend.WARP:

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            # Poiseuille flow profile: parabolic velocity distribution
            y = wp.float32(index[1])
            z = wp.float32(index[2])

            # Calculate normalized distance from center
            y_center = y - (H_y / 2.0)
            z_center = z - (H_z / 2.0)
            r_squared = (2.0 * y_center / H_y) ** 2.0 + (2.0 * z_center / H_z) ** 2.0

            # Parabolic profile: u = u_max * (1 - r²)
            return wp.vec(u_max * wp.max(0.0, 1.0 - r_squared), length=1)

        return bc_profile_warp


# Initialize Boundary Conditions
bc_left = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
# Alternatively, use a prescribed velocity profile
# bc_left = RegularizedBC("velocity", prescribed_value=(u_max, 0.0, 0.0), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)
bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_sphere = InterpolatedBounceBackSphereBC(indices=sphere, sphere_c=sphere_center, sphere_r=sphere_radius, u_wall=(0, 0, 0))
#bc_sphere = GradsApproximationBC(indices=sphere)
boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_sphere]

# Set up Momentum Transfer for Force Calculation
momentum_transfer = MomentumTransfer(bc_sphere, compute_backend=compute_backend)

# Setup Stepper
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="BGK",
)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D3Q19(
        precision_policy=precision_policy, compute_backend=ComputeBackend.JAX
    ),
)


def plot_drag_coefficient(time_steps, drag_coefficients,i,meta):
    """
    Plot the drag coefficient with various moving averages.

    Args:
        time_steps (list): List of time steps.
        drag_coefficients (list): List of drag coefficients.
    """
    # Convert lists to numpy arrays for processing
    time_steps_np = np.array(time_steps)
    drag_coefficients_np = np.array(drag_coefficients)

    # Define moving average windows
    windows = [10, 100, 1000, 10000, 100000]
    labels = ["MA 10", "MA 100", "MA 1,000", "MA 10,000", "MA 100,000"]

    plt.figure(figsize=(12, 8))
    plt.plot(time_steps_np, drag_coefficients_np, label="Raw", alpha=0.5)

    for window, label in zip(windows, labels):
        if len(drag_coefficients_np) >= window:
            ma = np.convolve(
                drag_coefficients_np, np.ones(window) / window, mode="valid"
            )
            plt.plot(time_steps_np[window - 1 :], ma, label=label)

    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Drag coefficient")
    plt.title("Drag Coefficient Over Time with Moving Averages")
    plt.savefig("drag_coefficient_ma{}.png".format(i))
    plt.close()


# Post-Processing Function
def post_process(
    step,
    f_current,
    drag_coefficients,
    lift_coefficients,
    time_steps,
    i,
    meta,
):
    # Convert to JAX array if necessary
    if not isinstance(f_current, jnp.ndarray):
        f_current = wp.to_jax(f_current)

    rho, u = macro(f_current)

    # Remove boundary cells
    u = u[:, 1:-1, 1:-1, 1:-1]
    rho = rho[:, 1:-1, 1:-1, 1:-1]
    u_magnitude = jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

    fields = {
        "rho": rho[0],
        "u_x": u[0],
        "u_y": u[1],
        "u_z": u[2],
        "u_magnitude": u_magnitude,
    }

    # Save the u_magnitude slice at the mid y-plane
    """     save_fields_vtk(fields, timestep=step)
    save_image(fields["u_magnitude"][:, grid_shape[1] // 2, :], timestep=step)
    print(
        f"Post-processed step {step}: Saved u_magnitude slice at y={grid_shape[1] // 2}"
    ) """

    # Compute lift and drag
    boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
    drag = boundary_force[0]  # x-direction
    lift = boundary_force[2]
    cd = 2.0 * drag / (u_max**2 * np.pi * sphere_radius**2)
    cl = 2.0 * lift / (u_max**2 * np.pi * sphere_radius**2)
    drag_coefficients.append(cd)
    lift_coefficients.append(cl)
    time_steps.append(step)
    # Plot drag coefficient
    plot_drag_coefficient(time_steps, drag_coefficients,i,meta)

    return np.mean(drag_coefficients)
# -------------------------- Simulation Loop --------------------------
# Initialize Lists to Store Coefficients and Time Steps
n = 10
Rerange = np.geomspace(1,200,n)
visc = u_max * sphere_radius / Rerange
omega = 1.0 / (3.0 * visc + 0.5)
meta = []


for i in range(n):
    time_steps = []
    drag_coefficients = []
    lift_coefficients = []

    start_time = time.time()
    for step in range(int(num_steps)):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, float(omega[i]), step)
        f_0, f_1 = f_1, f_0  # Swap the buffers

        if step % post_process_interval == 0 or step == num_steps - 1:
            cd = post_process(step, f_0, drag_coefficients,lift_coefficients,time_steps,i,meta)
            end_time = time.time()
            elapsed = end_time - start_time
            print(
                f"Completed step {step}. Time elapsed for {post_process_interval} steps: {elapsed:.6f} seconds."
            )
            start_time = time.time()
    meta.append(cd)

plt.figure(figsize=(12, 8))
plt.semilogx(Rerange, meta)
plt.xlabel("Re")
plt.ylabel("Drag coefficient")
plt.savefig("drag_coefficient_re.png".format(i))
plt.close()

meta = np.array(meta)

np.savez('./drag.npz',Rerange,meta)