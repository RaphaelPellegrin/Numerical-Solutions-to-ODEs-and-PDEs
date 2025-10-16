"""Finite Difference Methods for solving PDEs.

This module implements various numerical methods for solving partial differential equations:
1. 1D Heat Equation: ∂u/∂t = α ∂²u/∂x²
2. 2D Poisson Equation: ∇²u = f(x,y)

With comprehensive visualizations, animations, and stability analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
from pathlib import Path
from typing import Tuple, Optional, Callable


def solve_heat_equation_1d(
    alpha: float,
    L: float,
    T: float,
    nx: int,
    nt: int,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_conditions: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solves 1D heat equation using explicit finite differences.

    Solves: ∂u/∂t = α ∂²u/∂x² on domain [0,L] × [0,T]
    Using explicit scheme: u[i,n+1] = u[i,n] + r*(u[i+1,n] - 2*u[i,n] + u[i-1,n])
    where r = α*dt/dx² (stability requires r ≤ 0.5)

    Args:
        alpha: Thermal diffusivity coefficient
        L: Spatial domain length
        T: Total time
        nx: Number of spatial grid points
        nt: Number of time steps
        initial_condition: Function defining u(x,0)
        boundary_conditions: Tuple of (u(0,t), u(L,t))

    Returns:
        Tuple of (x_grid, t_grid, solution_matrix)
    """
    # Grid setup
    dx: float = L / (nx - 1)
    dt: float = T / nt
    r: float = alpha * dt / (dx**2)

    # Stability check
    if r > 0.5:
        print(f"Warning: r = {r:.4f} > 0.5, scheme may be unstable!")

    # Initialize grids
    x: np.ndarray = np.linspace(0, L, nx)
    t: np.ndarray = np.linspace(0, T, nt + 1)
    u: np.ndarray = np.zeros((nx, nt + 1))

    # Initial condition
    u[:, 0] = initial_condition(x)

    # Boundary conditions
    u[0, :] = boundary_conditions[0]
    u[-1, :] = boundary_conditions[1]

    # Time stepping
    for n in range(nt):
        for i in range(1, nx - 1):
            u[i, n + 1] = u[i, n] + r * (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n])

    return x, t, u


def solve_poisson_2d_jacobi(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    boundary_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[float], int]:
    """Solves 2D Poisson equation using Jacobi iteration.

    Solves: ∇²u = ∂²u/∂x² + ∂²u/∂y² = f(x,y)
    Using Jacobi iteration: u[i,j]^(k+1) = (u[i+1,j]^k + u[i-1,j]^k + u[i,j+1]^k + u[i,j-1]^k - h²f[i,j]) / 4

    Args:
        f: Source function f(x,y)
        boundary_func: Function defining boundary conditions
        Lx, Ly: Domain dimensions
        nx, ny: Grid points in x and y directions
        max_iter: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (x_grid, y_grid, solution, residual_history, iterations)
    """
    # Grid setup
    dx: float = Lx / (nx - 1)
    dy: float = Ly / (ny - 1)
    x: np.ndarray = np.linspace(0, Lx, nx)
    y: np.ndarray = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize solution and source term
    u: np.ndarray = np.zeros((ny, nx))
    u_new: np.ndarray = np.zeros((ny, nx))
    source: np.ndarray = f(X, Y)

    # Apply boundary conditions
    u[0, :] = boundary_func(X[0, :], Y[0, :])  # Bottom
    u[-1, :] = boundary_func(X[-1, :], Y[-1, :])  # Top
    u[:, 0] = boundary_func(X[:, 0], Y[:, 0])  # Left
    u[:, -1] = boundary_func(X[:, -1], Y[:, -1])  # Right

    residual_history: list[float] = []

    # Jacobi iteration
    for iteration in range(max_iter):
        u_new[:] = u[:]  # Copy current solution

        # Update interior points
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u_new[i, j] = 0.25 * (
                    u[i + 1, j]
                    + u[i - 1, j]
                    + u[i, j + 1]
                    + u[i, j - 1]
                    - dx**2 * source[i, j]
                )

        # Calculate residual
        residual: float = np.max(np.abs(u_new - u))
        residual_history.append(residual)

        # Check convergence
        if residual < tolerance:
            print(f"Jacobi converged in {iteration + 1} iterations")
            break

        u[:] = u_new[:]

    return x, y, u, residual_history, iteration + 1


def solve_poisson_2d_gauss_seidel(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    boundary_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[float], int]:
    """Solves 2D Poisson equation using Gauss-Seidel iteration.

    Similar to Jacobi but uses updated values immediately:
    u[i,j]^(k+1) = (u[i+1,j]^k + u[i-1,j]^(k+1) + u[i,j+1]^k + u[i,j-1]^(k+1) - h²f[i,j]) / 4

    Args:
        f: Source function f(x,y)
        boundary_func: Function defining boundary conditions
        Lx, Ly: Domain dimensions
        nx, ny: Grid points in x and y directions
        max_iter: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (x_grid, y_grid, solution, residual_history, iterations)
    """
    # Grid setup
    dx: float = Lx / (nx - 1)
    dy: float = Ly / (ny - 1)
    x: np.ndarray = np.linspace(0, Lx, nx)
    y: np.ndarray = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize solution and source term
    u: np.ndarray = np.zeros((ny, nx))
    u_old: np.ndarray = np.zeros((ny, nx))
    source: np.ndarray = f(X, Y)

    # Apply boundary conditions
    u[0, :] = boundary_func(X[0, :], Y[0, :])  # Bottom
    u[-1, :] = boundary_func(X[-1, :], Y[-1, :])  # Top
    u[:, 0] = boundary_func(X[:, 0], Y[:, 0])  # Left
    u[:, -1] = boundary_func(X[:, -1], Y[:, -1])  # Right

    residual_history: list[float] = []

    # Gauss-Seidel iteration
    for iteration in range(max_iter):
        u_old[:] = u[:]  # Store old solution for residual calculation

        # Update interior points (in-place)
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = 0.25 * (
                    u[i + 1, j]
                    + u[i - 1, j]
                    + u[i, j + 1]
                    + u[i, j - 1]
                    - dx**2 * source[i, j]
                )

        # Calculate residual
        residual: float = np.max(np.abs(u - u_old))
        residual_history.append(residual)

        # Check convergence
        if residual < tolerance:
            print(f"Gauss-Seidel converged in {iteration + 1} iterations")
            break

    return x, y, u, residual_history, iteration + 1


def create_heat_equation_gif(
    alpha: float,
    L: float,
    T: float,
    nx: int,
    nt: int,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_conditions: Tuple[float, float] = (0.0, 0.0),
    output_filename: str = "heat_equation_evolution.gif",
) -> None:
    """Creates animated GIF showing heat equation evolution over time.

    Args:
        alpha: Thermal diffusivity coefficient
        L: Spatial domain length
        T: Total time
        nx: Number of spatial grid points
        nt: Number of time steps
        initial_condition: Function defining u(x,0)
        boundary_conditions: Tuple of (u(0,t), u(L,t))
        output_filename: Name of output GIF file
    """
    # Solve heat equation
    x, t, u = solve_heat_equation_1d(
        alpha, L, T, nx, nt, initial_condition, boundary_conditions
    )

    # Create frames
    frames = []
    n_frames = min(50, nt + 1)  # Limit frames for reasonable file size
    frame_indices = np.linspace(0, nt, n_frames, dtype=int)

    for i, n in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot current solution
        ax.plot(x, u[:, n], "b-", linewidth=2, label=f"t = {t[n]:.3f}")

        # Plot initial condition for reference
        if n > 0:
            ax.plot(x, u[:, 0], "r--", alpha=0.5, label="Initial condition")

        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"1D Heat Equation Evolution (α={alpha}, Frame {i+1}/{n_frames})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(np.min(u) * 1.1, np.max(u) * 1.1)

        # Save frame
        plt.tight_layout()
        plt.savefig(f"temp_heat_frame_{i:03d}.png", dpi=100, bbox_inches="tight")
        frames.append(Image.open(f"temp_heat_frame_{i:03d}.png"))
        plt.close()

    # Create GIF
    frames[0].save(
        output_filename, save_all=True, append_images=frames[1:], duration=300, loop=0
    )

    # Clean up
    for i in range(n_frames):
        os.remove(f"temp_heat_frame_{i:03d}.png")

    print(f"Heat equation animation saved as {output_filename}")


def create_heat_lattice_animation(
    alpha: float,
    L: float,
    T: float,
    nx: int,
    nt: int,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_conditions: Tuple[float, float] = (0.0, 0.0),
    output_filename: str = "heat_lattice_march.gif",
) -> None:
    """Creates animated GIF showing heat equation lattice being built.

    Shows the space-time grid with computed points highlighted as we march forward in time.
    Grey points are uncomputed, blue points are computed.

    Args:
        alpha: Thermal diffusivity coefficient
        L: Spatial domain length
        T: Total time
        nx: Number of spatial grid points
        nt: Number of time steps
        initial_condition: Function defining u(x,0)
        boundary_conditions: Tuple of (u(0,t), u(L,t))
        output_filename: Name of output GIF file
    """
    # Solve heat equation
    x, t, u = solve_heat_equation_1d(
        alpha, L, T, nx, nt, initial_condition, boundary_conditions
    )

    # Create space-time meshgrid for visualization
    X, T_grid = np.meshgrid(x, t)

    frames = []
    n_frames = min(30, nt + 1)
    frame_indices = np.linspace(0, nt, n_frames, dtype=int)

    for frame_idx, current_time_idx in enumerate(frame_indices):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Space-time lattice
        # Plot all grid points as grey initially
        ax1.scatter(X.flatten(), T_grid.flatten(), c="lightgrey", s=20, alpha=0.5)

        # Highlight computed points (up to current time) in blue
        for t_idx in range(current_time_idx + 1):
            ax1.scatter(x, t[t_idx] * np.ones_like(x), c="blue", s=30)

        # Highlight current time level in red
        if current_time_idx > 0:
            ax1.scatter(x, t[current_time_idx] * np.ones_like(x), c="red", s=50)

        # Show stencil for interior points at current time
        if current_time_idx > 0:
            for i in range(1, nx - 1, max(1, nx // 10)):  # Show every few points
                # Stencil points: (i-1,n), (i,n), (i+1,n), (i,n+1)
                stencil_x = [x[i - 1], x[i], x[i + 1], x[i]]
                stencil_t = [
                    t[current_time_idx - 1],
                    t[current_time_idx - 1],
                    t[current_time_idx - 1],
                    t[current_time_idx],
                ]
                ax1.plot(stencil_x, stencil_t, "g-", linewidth=2, alpha=0.7)

        ax1.set_xlabel("Space (x)")
        ax1.set_ylabel("Time (t)")
        ax1.set_title(
            f"Heat Equation Lattice March\n(Time step {current_time_idx}/{nt})"
        )
        ax1.grid(True, alpha=0.3)

        # Right plot: Current solution profile
        ax2.plot(
            x,
            u[:, current_time_idx],
            "b-",
            linewidth=2,
            label=f"t = {t[current_time_idx]:.3f}",
        )
        ax2.plot(x, u[:, 0], "r--", alpha=0.5, label="Initial condition")
        ax2.set_xlabel("x")
        ax2.set_ylabel("u(x,t)")
        ax2.set_title("Current Temperature Profile")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(np.min(u) * 1.1, np.max(u) * 1.1)

        plt.tight_layout()
        plt.savefig(
            f"temp_lattice_frame_{frame_idx:03d}.png", dpi=100, bbox_inches="tight"
        )
        frames.append(Image.open(f"temp_lattice_frame_{frame_idx:03d}.png"))
        plt.close()

    # Create GIF
    frames[0].save(
        output_filename, save_all=True, append_images=frames[1:], duration=500, loop=0
    )

    # Clean up
    for i in range(n_frames):
        os.remove(f"temp_lattice_frame_{i:03d}.png")

    print(f"Heat lattice animation saved as {output_filename}")


def create_poisson_iteration_gif(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    boundary_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    method: str = "jacobi",
    max_iter: int = 100,
    output_filename: str = "poisson_iteration.gif",
) -> None:
    """Creates animated GIF showing Poisson equation iterative solution convergence.

    Args:
        f: Source function f(x,y)
        boundary_func: Function defining boundary conditions
        Lx, Ly: Domain dimensions
        nx, ny: Grid points in x and y directions
        method: "jacobi" or "gauss_seidel"
        max_iter: Maximum iterations to show
        output_filename: Name of output GIF file
    """
    # Modified solver to return intermediate solutions
    dx: float = Lx / (nx - 1)
    dy: float = Ly / (ny - 1)
    x: np.ndarray = np.linspace(0, Lx, nx)
    y: np.ndarray = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    u: np.ndarray = np.zeros((ny, nx))
    source: np.ndarray = f(X, Y)

    # Apply boundary conditions
    u[0, :] = boundary_func(X[0, :], Y[0, :])
    u[-1, :] = boundary_func(X[-1, :], Y[-1, :])
    u[:, 0] = boundary_func(X[:, 0], Y[:, 0])
    u[:, -1] = boundary_func(X[:, -1], Y[:, -1])

    frames = []
    solutions = [u.copy()]
    residuals = []

    # Perform iterations and store intermediate results
    for iteration in range(max_iter):
        u_old = u.copy()

        if method.lower() == "jacobi":
            u_new = u.copy()
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    u_new[i, j] = 0.25 * (
                        u[i + 1, j]
                        + u[i - 1, j]
                        + u[i, j + 1]
                        + u[i, j - 1]
                        - dx**2 * source[i, j]
                    )
            u[:] = u_new[:]
        else:  # Gauss-Seidel
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    u[i, j] = 0.25 * (
                        u[i + 1, j]
                        + u[i - 1, j]
                        + u[i, j + 1]
                        + u[i, j - 1]
                        - dx**2 * source[i, j]
                    )

        residual = np.max(np.abs(u - u_old))
        residuals.append(residual)
        solutions.append(u.copy())

        if residual < 1e-6:
            break

    # Calculate global min/max for consistent colorbar
    all_solutions = np.array(solutions)
    vmin, vmax = np.min(all_solutions), np.max(all_solutions)
    if vmax - vmin < 1e-10:  # Handle case where solution is nearly zero
        vmin, vmax = -0.1, 0.1

    # Create consistent levels for all frames
    levels = np.linspace(vmin, vmax, 21)

    # Create animation frames
    n_frames = min(30, len(solutions))
    frame_indices = np.linspace(0, len(solutions) - 1, n_frames, dtype=int)

    for frame_idx, sol_idx in enumerate(frame_indices):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left: Solution contour plot with consistent colorbar
        im = ax1.contourf(
            X,
            Y,
            solutions[sol_idx],
            levels=levels,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax1.contour(
            X,
            Y,
            solutions[sol_idx],
            levels=levels,
            colors="black",
            alpha=0.3,
            linewidths=0.5,
        )
        plt.colorbar(im, ax=ax1)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title(
            f"{method.title()} Iteration {sol_idx}\n(Max residual: {residuals[sol_idx-1]:.2e})"
            if sol_idx > 0
            else f"{method.title()} Initial Guess"
        )
        ax1.set_aspect("equal")

        # Right: Residual convergence
        if len(residuals) > 0:
            ax2.semilogy(range(1, len(residuals) + 1), residuals, "b-", linewidth=2)
            if sol_idx > 0:
                ax2.semilogy(sol_idx, residuals[sol_idx - 1], "ro", markersize=8)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Max Residual")
        ax2.set_title("Convergence History")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"temp_poisson_frame_{frame_idx:03d}.png", dpi=100, bbox_inches="tight"
        )
        frames.append(Image.open(f"temp_poisson_frame_{frame_idx:03d}.png"))
        plt.close()

    # Create GIF
    frames[0].save(
        output_filename, save_all=True, append_images=frames[1:], duration=400, loop=0
    )

    # Clean up
    for i in range(n_frames):
        os.remove(f"temp_poisson_frame_{i:03d}.png")

    print(f"Poisson iteration animation saved as {output_filename}")


def create_grid_update_animation(
    nx: int = 21, ny: int = 21, output_filename: str = "grid_stencil_animation.gif"
) -> None:
    """Creates animated GIF showing grid point-by-point update with stencil and solution.

    Shows how each point in a 2D grid is updated using neighboring points in the
    finite difference stencil for Poisson equation, with color-coded solution background.

    Args:
        nx, ny: Grid dimensions
        output_filename: Name of output GIF file
    """
    # Create grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Define a test Poisson problem: ∇²u = -2π²sin(πx)sin(πy)
    # Analytical solution: u = sin(πx)sin(πy)
    def source_function(X, Y):
        return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    def boundary_zero(X, Y):
        return np.zeros_like(X)

    # Initialize solution
    u = np.zeros((ny, nx))
    dx = 1.0 / (nx - 1)
    source = source_function(X, Y)

    # Apply boundary conditions (u = 0 on boundaries)
    u[0, :] = boundary_zero(X[0, :], Y[0, :])
    u[-1, :] = boundary_zero(X[-1, :], Y[-1, :])
    u[:, 0] = boundary_zero(X[:, 0], Y[:, 0])
    u[:, -1] = boundary_zero(X[:, -1], Y[:, -1])

    # Get all interior points in order (row by row)
    interior_points = []
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            interior_points.append((i, j))

    # Perform MASSIVE number of sweeps for full convergence (1M+ iterations)
    # Jacobi method is very slow - need many more iterations for fine grids
    min_iterations = 1000000  # At least 1 million iterations
    n_sweeps = max(
        500, min_iterations // len(interior_points)
    )  # Ensure at least 1M iterations
    total_iterations = n_sweeps * len(interior_points)

    # Create frames showing key convergence milestones
    max_frames = 200  # More frames to capture the slow convergence
    frame_step = max(1, total_iterations // max_frames)

    selected_iterations = list(range(0, total_iterations, frame_step))

    frames = []

    # Calculate global min/max for consistent colorbar
    # Do a few iterations to get reasonable range
    u_temp = u.copy()
    for _ in range(10):
        for i, j in interior_points:
            u_temp[i, j] = 0.25 * (
                u_temp[i + 1, j]
                + u_temp[i - 1, j]
                + u_temp[i, j + 1]
                + u_temp[i, j - 1]
                - dx**2 * source[i, j]
            )

    vmin, vmax = np.min(u_temp), np.max(u_temp)
    if vmax - vmin < 1e-10:  # Handle case where solution is nearly zero
        vmin, vmax = -0.1, 0.1

    for frame_idx, iteration in enumerate(selected_iterations):
        # Determine which point we're updating in this iteration
        sweep_num = iteration // len(interior_points)
        point_in_sweep = iteration % len(interior_points)
        i, j = interior_points[point_in_sweep]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Left plot: Solution with stencil overlay
        levels = np.linspace(vmin, vmax, 21)
        im = ax1.contourf(
            X, Y, u, levels=levels, cmap="RdYlBu_r", alpha=0.8, vmin=vmin, vmax=vmax
        )
        ax1.contour(X, Y, u, levels=levels, colors="black", alpha=0.3, linewidths=0.5)

        # Plot grid points
        ax1.scatter(X.flatten(), Y.flatten(), c="black", s=15, alpha=0.4)

        # Highlight boundary points
        boundary_mask = np.zeros_like(X, dtype=bool)
        boundary_mask[0, :] = True  # Bottom
        boundary_mask[-1, :] = True  # Top
        boundary_mask[:, 0] = True  # Left
        boundary_mask[:, -1] = True  # Right

        ax1.scatter(
            X[boundary_mask],
            Y[boundary_mask],
            c="red",
            s=40,
            marker="s",
            alpha=0.8,
            label="Boundary points",
        )

        # Highlight current point being updated
        ax1.scatter(
            X[i, j],
            Y[i, j],
            c="blue",
            s=200,
            marker="s",
            edgecolor="white",
            linewidth=2,
            label=f"Current point ({i},{j})",
        )

        # Highlight stencil points
        stencil_points = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        for si, sj in stencil_points:
            ax1.scatter(
                X[si, sj],
                Y[si, sj],
                c="lime",
                s=120,
                marker="^",
                edgecolor="darkgreen",
                linewidth=1,
                alpha=0.9,
            )

        # Draw stencil connections
        ax1.plot(
            [X[i - 1, j], X[i, j]],
            [Y[i - 1, j], Y[i, j]],
            "lime",
            linewidth=4,
            alpha=0.8,
        )
        ax1.plot(
            [X[i + 1, j], X[i, j]],
            [Y[i + 1, j], Y[i, j]],
            "lime",
            linewidth=4,
            alpha=0.8,
        )
        ax1.plot(
            [X[i, j - 1], X[i, j]],
            [Y[i, j - 1], Y[i, j]],
            "lime",
            linewidth=4,
            alpha=0.8,
        )
        ax1.plot(
            [X[i, j + 1], X[i, j]],
            [Y[i, j + 1], Y[i, j]],
            "lime",
            linewidth=4,
            alpha=0.8,
        )

        # Update the current point using Jacobi iteration
        old_value = u[i, j]
        u[i, j] = 0.25 * (
            u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - dx**2 * source[i, j]
        )

        # Check convergence every complete sweep
        if point_in_sweep == len(interior_points) - 1:  # End of sweep
            max_change = (
                np.max(np.abs(u - u_temp)) if "u_temp" in locals() else float("inf")
            )
            if max_change < 1e-10:  # Converged to machine precision
                print(
                    f"Converged after {iteration+1} iterations (max change: {max_change:.2e})"
                )

        # Store solution at beginning of each sweep for convergence check
        if point_in_sweep == 0:
            u_temp = u.copy()

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title(
            f"Poisson Equation: 5-Point Stencil Update\nSweep {sweep_num+1}/{n_sweeps}, Point ({i},{j}) - Iteration {iteration+1}/{total_iterations}"
        )
        ax1.legend(loc="upper left", bbox_to_anchor=(0, 1))
        ax1.set_aspect("equal")
        plt.colorbar(im, ax=ax1, shrink=0.8)

        # Right plot: Stencil values and calculation
        ax2.axis("off")

        # Create a zoomed view of the stencil
        stencil_info = f"""Jacobi Iteration Progress:

Sweep: {sweep_num+1}/{n_sweeps}
Point: ({i},{j}) - {point_in_sweep+1}/{len(interior_points)} in sweep
Total iteration: {iteration+1}/{total_iterations}

Current point: u[{i},{j}] = {old_value:.4f}

Stencil values:
  u[{i-1},{j}] = {u[i-1, j]:.4f}  (North)
  u[{i+1},{j}] = {u[i+1, j]:.4f}  (South)  
  u[{i},{j-1}] = {u[i, j-1]:.4f}  (West)
  u[{i},{j+1}] = {u[i, j+1]:.4f}  (East)

Source: f[{i},{j}] = {source[i, j]:.4f}
Grid spacing: h = {dx:.4f}

Update formula:
u_new = (u_N + u_S + u_W + u_E - h²f) / 4

u_new[{i},{j}] = ({u[i-1, j]:.4f} + {u[i+1, j]:.4f} + {u[i, j-1]:.4f} + {u[i, j+1]:.4f} - {dx**2 * source[i, j]:.4f}) / 4

u_new[{i},{j}] = {u[i, j]:.4f}

Change: Δu = {u[i, j] - old_value:.4f}"""

        ax2.text(
            0.05,
            0.95,
            stencil_info,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # Add a small diagram of the stencil
        ax2.text(
            0.05,
            0.35,
            "Stencil Pattern:",
            transform=ax2.transAxes,
            fontsize=12,
            weight="bold",
        )

        # Draw stencil diagram
        center_x, center_y = 0.3, 0.2
        spacing = 0.08

        # Draw points
        ax2.plot(
            center_x, center_y, "bs", markersize=15, transform=ax2.transAxes
        )  # Center
        ax2.plot(
            center_x, center_y + spacing, "g^", markersize=12, transform=ax2.transAxes
        )  # North
        ax2.plot(
            center_x, center_y - spacing, "g^", markersize=12, transform=ax2.transAxes
        )  # South
        ax2.plot(
            center_x - spacing, center_y, "g^", markersize=12, transform=ax2.transAxes
        )  # West
        ax2.plot(
            center_x + spacing, center_y, "g^", markersize=12, transform=ax2.transAxes
        )  # East

        # Draw connections
        ax2.plot(
            [center_x, center_x],
            [center_y - spacing, center_y + spacing],
            "g-",
            linewidth=3,
            alpha=0.7,
            transform=ax2.transAxes,
        )
        ax2.plot(
            [center_x - spacing, center_x + spacing],
            [center_y, center_y],
            "g-",
            linewidth=3,
            alpha=0.7,
            transform=ax2.transAxes,
        )

        # Labels
        ax2.text(
            center_x,
            center_y - 0.04,
            f"({i},{j})",
            transform=ax2.transAxes,
            ha="center",
            fontsize=8,
        )

        # Ensure consistent layout
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        plt.savefig(
            f"temp_stencil_frame_{frame_idx:03d}.png", dpi=100, bbox_inches="tight"
        )
        frames.append(Image.open(f"temp_stencil_frame_{frame_idx:03d}.png"))
        plt.close()

    # Create GIF
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=25,  # Ultra fast animation - 25ms per frame (40 fps)
        loop=0,
    )

    # Clean up
    for i in range(len(selected_iterations)):
        os.remove(f"temp_stencil_frame_{i:03d}.png")

    print(f"Enhanced grid stencil animation saved as {output_filename}")
    print(
        f"Animation shows {len(selected_iterations)} frames over {n_sweeps} complete sweeps"
    )
    print(
        f"Total iterations: {total_iterations:,} (showing every {frame_step} iterations)"
    )
    print(
        f"Each sweep processes all {len(interior_points)} interior points sequentially"
    )
    print(f"Stencil moves point-by-point: row by row, left to right")
    print(
        f"Animation speed: 25ms per frame (40 fps) for ultra-fast convergence visualization"
    )
    print(
        f"WARNING: This will run {total_iterations:,} iterations - may take several minutes to generate!"
    )


def plot_heat_results(
    alpha: float,
    L: float,
    T: float,
    nx: int,
    nt: int,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_conditions: Tuple[float, float] = (0.0, 0.0),
) -> None:
    """Plots comprehensive results for heat equation.

    Creates multiple subplots showing solution evolution, stability analysis,
    and various cross-sections.
    """
    # Solve heat equation
    x, t, u = solve_heat_equation_1d(
        alpha, L, T, nx, nt, initial_condition, boundary_conditions
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"1D Heat Equation Analysis (α={alpha})", fontsize=16)

    # 1. Space-time contour plot
    ax1 = axes[0, 0]
    X, T_grid = np.meshgrid(x, t)
    im1 = ax1.contourf(X, T_grid, u.T, levels=20, cmap="hot")
    ax1.contour(X, T_grid, u.T, levels=20, colors="black", alpha=0.3, linewidths=0.5)
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel("Space (x)")
    ax1.set_ylabel("Time (t)")
    ax1.set_title("Temperature Evolution")

    # 2. Solution at different times
    ax2 = axes[0, 1]
    time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt]
    colors = ["red", "orange", "green", "blue", "purple"]
    for i, (time_idx, color) in enumerate(zip(time_indices, colors)):
        ax2.plot(
            x, u[:, time_idx], color=color, linewidth=2, label=f"t = {t[time_idx]:.3f}"
        )
    ax2.set_xlabel("x")
    ax2.set_ylabel("u(x,t)")
    ax2.set_title("Temperature Profiles")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Temperature evolution at specific points
    ax3 = axes[0, 2]
    x_indices = [nx // 4, nx // 2, 3 * nx // 4]
    for i, x_idx in enumerate(x_indices):
        ax3.plot(t, u[x_idx, :], linewidth=2, label=f"x = {x[x_idx]:.3f}")
    ax3.set_xlabel("Time (t)")
    ax3.set_ylabel("u(x,t)")
    ax3.set_title("Temperature vs Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Stability analysis
    ax4 = axes[1, 0]
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / (dx**2)

    # Plot stability region
    r_values = np.linspace(0, 1, 100)
    stability_bound = 0.5 * np.ones_like(r_values)
    ax4.plot(
        r_values, stability_bound, "r--", linewidth=2, label="Stability limit (r = 0.5)"
    )
    ax4.axvline(x=r, color="blue", linewidth=2, label=f"Current r = {r:.4f}")
    ax4.fill_between(
        r_values, 0, stability_bound, alpha=0.3, color="green", label="Stable region"
    )
    ax4.set_xlabel("r = α·dt/dx²")
    ax4.set_ylabel("Stability")
    ax4.set_title("Stability Analysis")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # 5. Energy evolution (L2 norm)
    ax5 = axes[1, 1]
    energy = np.array([np.trapz(u[:, n] ** 2, x) for n in range(nt + 1)])
    ax5.plot(t, energy, "b-", linewidth=2)
    ax5.set_xlabel("Time (t)")
    ax5.set_ylabel("Energy (∫u²dx)")
    ax5.set_title("Energy Evolution")
    ax5.grid(True, alpha=0.3)

    # 6. Maximum temperature evolution
    ax6 = axes[1, 2]
    max_temp = np.array([np.max(u[:, n]) for n in range(nt + 1)])
    min_temp = np.array([np.min(u[:, n]) for n in range(nt + 1)])
    ax6.plot(t, max_temp, "r-", linewidth=2, label="Maximum")
    ax6.plot(t, min_temp, "b-", linewidth=2, label="Minimum")
    ax6.set_xlabel("Time (t)")
    ax6.set_ylabel("Temperature")
    ax6.set_title("Temperature Extremes")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("heat_equation_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_poisson_results(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    boundary_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
) -> None:
    """Plots comprehensive results for Poisson equation.

    Compares Jacobi and Gauss-Seidel methods with convergence analysis.
    """
    # Solve with both methods
    x_j, y_j, u_jacobi, res_j, iter_j = solve_poisson_2d_jacobi(
        f, boundary_func, Lx, Ly, nx, ny, max_iter=1000
    )
    x_gs, y_gs, u_gauss, res_gs, iter_gs = solve_poisson_2d_gauss_seidel(
        f, boundary_func, Lx, Ly, nx, ny, max_iter=1000
    )

    X, Y = np.meshgrid(x_j, y_j)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("2D Poisson Equation Analysis", fontsize=16)

    # 1. Jacobi solution
    ax1 = axes[0, 0]
    im1 = ax1.contourf(X, Y, u_jacobi, levels=20, cmap="viridis")
    ax1.contour(X, Y, u_jacobi, levels=20, colors="black", alpha=0.3, linewidths=0.5)
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"Jacobi Solution ({iter_j} iterations)")
    ax1.set_aspect("equal")

    # 2. Gauss-Seidel solution
    ax2 = axes[0, 1]
    im2 = ax2.contourf(X, Y, u_gauss, levels=20, cmap="viridis")
    ax2.contour(X, Y, u_gauss, levels=20, colors="black", alpha=0.3, linewidths=0.5)
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"Gauss-Seidel Solution ({iter_gs} iterations)")
    ax2.set_aspect("equal")

    # 3. Difference between methods
    ax3 = axes[0, 2]
    diff = np.abs(u_jacobi - u_gauss)
    im3 = ax3.contourf(X, Y, diff, levels=20, cmap="Reds")
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("|Jacobi - Gauss-Seidel|")
    ax3.set_aspect("equal")

    # 4. Convergence comparison
    ax4 = axes[1, 0]
    ax4.semilogy(range(1, len(res_j) + 1), res_j, "b-", linewidth=2, label="Jacobi")
    ax4.semilogy(
        range(1, len(res_gs) + 1), res_gs, "r-", linewidth=2, label="Gauss-Seidel"
    )
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Max Residual")
    ax4.set_title("Convergence Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Cross-section comparison
    ax5 = axes[1, 1]
    mid_y = ny // 2
    ax5.plot(x_j, u_jacobi[mid_y, :], "b-", linewidth=2, label="Jacobi")
    ax5.plot(x_gs, u_gauss[mid_y, :], "r--", linewidth=2, label="Gauss-Seidel")
    ax5.set_xlabel("x")
    ax5.set_ylabel("u(x, y_mid)")
    ax5.set_title(f"Cross-section at y = {y_j[mid_y]:.2f}")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 3D surface plot
    ax6 = axes[1, 2]
    ax6.remove()  # Remove 2D axis
    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    surf = ax6.plot_surface(X, Y, u_gauss, cmap="viridis", alpha=0.8)
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_zlabel("u(x,y)")
    ax6.set_title("3D Solution Surface")

    plt.tight_layout()
    plt.savefig("poisson_equation_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


# Example usage and test functions
if __name__ == "__main__":
    print("Running PDE solver demonstrations...")

    # Heat equation example
    print("\n1. Heat Equation Example")
    alpha = 0.01
    L = 1.0
    T = 0.5
    nx = 51
    nt = 1000

    # Initial condition: Gaussian pulse
    def initial_gaussian(x):
        return np.exp(-50 * (x - 0.5) ** 2)

    # Create heat equation animations
    create_heat_equation_gif(
        alpha, L, T, nx, nt, initial_gaussian, output_filename="heat_evolution.gif"
    )
    create_heat_lattice_animation(
        alpha, L, T, nx, nt, initial_gaussian, output_filename="heat_lattice.gif"
    )

    # Plot comprehensive results
    plot_heat_results(alpha, L, T, nx, nt, initial_gaussian)

    # Poisson equation example
    print("\n2. Poisson Equation Example")

    # Source function: f(x,y) = -2π²sin(πx)sin(πy)
    def source_function(X, Y):
        return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Boundary conditions: u = 0 on all boundaries
    def boundary_zero(X, Y):
        return np.zeros_like(X)

    Lx, Ly = 1.0, 1.0
    nx, ny = 41, 41

    # Create Poisson animations
    create_poisson_iteration_gif(
        source_function,
        boundary_zero,
        Lx,
        Ly,
        nx,
        ny,
        method="jacobi",
        output_filename="poisson_jacobi.gif",
    )
    create_poisson_iteration_gif(
        source_function,
        boundary_zero,
        Lx,
        Ly,
        nx,
        ny,
        method="gauss_seidel",
        output_filename="poisson_gauss_seidel.gif",
    )

    # Create stencil animation
    create_grid_update_animation(21, 21, "stencil_animation.gif")

    # Plot comprehensive results
    plot_poisson_results(source_function, boundary_zero, Lx, Ly, nx, ny)

    print("\nAll PDE demonstrations completed!")
    print("Generated files:")
    print("- heat_evolution.gif")
    print("- heat_lattice.gif")
    print("- poisson_jacobi.gif")
    print("- poisson_gauss_seidel.gif")
    print("- stencil_animation.gif")
    print("- heat_equation_analysis.png")
    print("- poisson_equation_analysis.png")
