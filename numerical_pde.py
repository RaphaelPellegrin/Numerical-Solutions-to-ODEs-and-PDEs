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
    """Creates animated GIF showing heat equation lattice being built with clear stencil visualization.

    Shows the space-time grid with computed points highlighted as we march forward in time.
    Uses a coarser grid for better stencil visibility and shows ALL stencils clearly.

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
    # Use coarser grid for better visualization if original is too fine
    nx_vis = min(nx, 15)  # Limit to 15 points for clear stencil visibility
    nt_vis = min(nt, 12)  # Limit time steps for clear visualization

    # Solve heat equation with coarser grid for visualization
    x_vis, t_vis, u_vis = solve_heat_equation_1d(
        alpha, L, T, nx_vis, nt_vis, initial_condition, boundary_conditions
    )

    # Create space-time meshgrid for visualization
    X_vis, T_vis_grid = np.meshgrid(x_vis, t_vis)

    frames: list[Image.Image] = []
    n_frames = min(20, nt_vis + 1)
    frame_indices = np.linspace(0, nt_vis, n_frames, dtype=int)

    for frame_idx, current_time_idx in enumerate(frame_indices):
        fig = plt.figure(figsize=(16, 10))

        # Create subplot layout: 2 plots on top, text box below
        ax1 = plt.subplot(2, 2, 1)  # Top left
        ax2 = plt.subplot(2, 2, 2)  # Top right
        ax3 = plt.subplot(2, 1, 2)  # Bottom (spans full width)
        ax3.axis("off")  # Turn off axis for text area

        # Left plot: Space-time lattice with enhanced stencil visualization
        # Plot all grid points as larger grey dots for better visibility
        ax1.scatter(
            X_vis.flatten(),
            T_vis_grid.flatten(),
            c="lightgrey",
            s=60,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.5,
        )

        # Highlight computed points (up to current time) in blue with larger size
        for t_idx in range(current_time_idx + 1):
            ax1.scatter(
                x_vis,
                t_vis[t_idx] * np.ones_like(x_vis),
                c="blue",
                s=80,
                alpha=0.8,
                edgecolor="darkblue",
                linewidth=1,
            )

        # Highlight current time level in red with even larger size
        if current_time_idx > 0:
            ax1.scatter(
                x_vis,
                t_vis[current_time_idx] * np.ones_like(x_vis),
                c="red",
                s=120,
                alpha=0.9,
                edgecolor="darkred",
                linewidth=2,
            )

        # Show ALL stencils for interior points at current time (not just a subset)
        if current_time_idx > 0:
            # Show stencils for ALL interior points to make the pattern clear
            for i in range(
                1, nx_vis - 1
            ):  # Show ALL interior points, not just a subset
                # Stencil points: (i-1,n), (i,n), (i+1,n) -> (i,n+1)
                stencil_x = [x_vis[i - 1], x_vis[i], x_vis[i + 1]]
                stencil_t = [
                    t_vis[current_time_idx - 1],
                    t_vis[current_time_idx - 1],
                    t_vis[current_time_idx - 1],
                ]
                target_x = x_vis[i]
                target_t = t_vis[current_time_idx]

                # Draw stencil connections with different colors for clarity
                # Connections from stencil points to target
                for sx, st in zip(stencil_x, stencil_t):
                    ax1.plot(
                        [sx, target_x], [st, target_t], "lime", linewidth=3, alpha=0.8
                    )

                # Highlight stencil points with special markers
                ax1.scatter(
                    stencil_x,
                    stencil_t,
                    c="orange",
                    s=100,
                    marker="^",
                    edgecolor="darkorange",
                    linewidth=2,
                    alpha=0.9,
                    zorder=5,
                )

                # Highlight target point being computed
                ax1.scatter(
                    target_x,
                    target_t,
                    c="yellow",
                    s=140,
                    marker="s",
                    edgecolor="gold",
                    linewidth=3,
                    alpha=0.9,
                    zorder=6,
                )

        # Add boundary points highlighting
        if current_time_idx >= 0:
            # Highlight boundary points at all computed time levels
            for t_idx in range(current_time_idx + 1):
                ax1.scatter(
                    [x_vis[0], x_vis[-1]],
                    [t_vis[t_idx], t_vis[t_idx]],
                    c="purple",
                    s=100,
                    marker="D",
                    edgecolor="darkviolet",
                    linewidth=2,
                    alpha=0.8,
                )

        ax1.set_xlabel("Space (x)", fontsize=12)
        ax1.set_ylabel("Time (t)", fontsize=12)
        ax1.set_title(
            f"Heat Equation Explicit Stencil Lattice\nTime step {current_time_idx}/{nt_vis} (Grid: {nx_vis}×{nt_vis+1})",
            fontsize=14,
        )
        ax1.grid(True, alpha=0.4, linewidth=1)
        ax1.set_xlim(-0.05, L + 0.05)
        ax1.set_ylim(-T * 0.05, T + T * 0.05)

        # Add legend for clarity
        legend_elements = [
            plt.scatter(
                [],
                [],
                c="lightgrey",
                s=60,
                alpha=0.6,
                edgecolor="black",
                label="Uncomputed points",
            ),
            plt.scatter(
                [],
                [],
                c="blue",
                s=80,
                alpha=0.8,
                edgecolor="darkblue",
                label="Computed points",
            ),
            plt.scatter(
                [],
                [],
                c="red",
                s=120,
                alpha=0.9,
                edgecolor="darkred",
                label="Current time level",
            ),
            plt.scatter(
                [],
                [],
                c="orange",
                s=100,
                marker="^",
                edgecolor="darkorange",
                label="Stencil points",
            ),
            plt.scatter(
                [],
                [],
                c="yellow",
                s=140,
                marker="s",
                edgecolor="gold",
                label="Target points",
            ),
            plt.scatter(
                [],
                [],
                c="purple",
                s=100,
                marker="D",
                edgecolor="darkviolet",
                label="Boundary points",
            ),
        ]
        ax1.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0, 1),
            fontsize=10,
        )

        # Right plot: Current solution profile with enhanced visualization
        ax2.plot(
            x_vis,
            u_vis[:, current_time_idx],
            "b-",
            linewidth=3,
            marker="o",
            markersize=8,
            label=f"t = {t_vis[current_time_idx]:.3f}",
        )
        ax2.plot(
            x_vis,
            u_vis[:, 0],
            "r--",
            alpha=0.7,
            linewidth=2,
            marker="s",
            markersize=6,
            label="Initial condition",
        )

        # Highlight boundary conditions
        ax2.scatter(
            [x_vis[0], x_vis[-1]],
            [u_vis[0, current_time_idx], u_vis[-1, current_time_idx]],
            c="purple",
            s=120,
            marker="D",
            edgecolor="darkviolet",
            linewidth=2,
            zorder=5,
            label="Boundary conditions",
        )

        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel("u(x,t)", fontsize=12)
        ax2.set_title("Temperature Profile", fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.4)
        ax2.set_ylim(np.min(u_vis) * 1.1, np.max(u_vis) * 1.1)

        # Add text box with stencil information
        if current_time_idx > 0:
            stencil_info = f"""Explicit Finite Difference Stencil:

Grid: {nx_vis} spatial × {nt_vis+1} temporal points
Current time step: {current_time_idx}/{nt_vis}

For each interior point u[i,n+1]:
u[i,n+1] = u[i,n] + r·(u[i+1,n] - 2·u[i,n] + u[i-1,n])

where r = α·dt/dx² = {alpha * (T/nt_vis) / ((L/(nx_vis-1))**2):.4f}

Stencil pattern (orange → yellow):
• 3 points at time n (triangles)
• 1 point at time n+1 (square)

ALL interior stencils shown for clarity!"""
        else:
            stencil_info = f"""Initial Condition Setup:

Grid: {nx_vis} spatial × {nt_vis+1} temporal points
Boundary conditions (purple diamonds):
• u(0,t) = {boundary_conditions[0]}
• u(L,t) = {boundary_conditions[1]}

Initial condition u(x,0) applied to all spatial points.
Ready to begin time marching with explicit scheme."""

        ax3.text(
            0.05,
            0.95,
            stencil_info,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcyan", alpha=0.9),
        )

        plt.tight_layout()
        plt.savefig(
            f"temp_lattice_frame_{frame_idx:03d}.png", dpi=120, bbox_inches="tight"
        )
        frames.append(Image.open(f"temp_lattice_frame_{frame_idx:03d}.png"))
        plt.close()

    # Create GIF with slower duration for better visibility
    frames[0].save(
        output_filename, save_all=True, append_images=frames[1:], duration=800, loop=0
    )

    # Clean up
    for i in range(n_frames):
        os.remove(f"temp_lattice_frame_{i:03d}.png")

    print(f"Enhanced heat lattice animation saved as {output_filename}")
    print(f"Coarse grid used: {nx_vis} spatial × {nt_vis+1} temporal points")
    print(f"ALL stencils shown for maximum clarity")
    print(f"Animation duration: 800ms per frame for detailed viewing")


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


def create_heat_stencil_animation(
    alpha: float = 0.01,
    L: float = 1.0,
    T: float = 1.0,  # Changed from 0.1 to 1.0
    nx: int = 21,
    nt: int = 20,
    initial_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    boundary_conditions: Tuple[float, float] = (0.0, 0.0),
    output_filename: str = "heat_stencil_animation.gif",
) -> None:
    """Creates animated GIF showing heat equation explicit finite difference stencil progression.

    Shows the stencil moving left-to-right across space, then advancing in time.
    Each point u[i,n+1] is computed using the stencil: u[i-1,n], u[i,n], u[i+1,n].

    Args:
        alpha: Thermal diffusivity coefficient
        L: Spatial domain length
        T: Total time
        nx: Number of spatial grid points
        nt: Number of time steps
        initial_condition: Function defining u(x,0), defaults to Gaussian pulse
        boundary_conditions: Tuple of (u(0,t), u(L,t))
        output_filename: Name of output GIF file
    """
    # Default initial condition if none provided
    if initial_condition is None:

        def initial_condition(x):
            return np.exp(-50 * (x - 0.5) ** 2)

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

    # Create space-time meshgrid for visualization
    X_space, T_grid = np.meshgrid(x, t)

    frames: list[Image.Image] = []

    # For each time step (except the first which is initial condition)
    for n in range(nt):
        # For each interior spatial point (left to right)
        for i in range(1, nx - 1):
            fig = plt.figure(figsize=(16, 12))

            # Create subplot layout: 2 plots on top, text box below
            ax1 = plt.subplot(2, 2, 1)  # Top left
            ax2 = plt.subplot(2, 2, 2)  # Top right
            ax3 = plt.subplot(2, 1, 2)  # Bottom (spans full width)
            ax3.axis("off")  # Turn off axis for text area

            # Left plot: Space-time lattice with stencil
            # Plot all grid points as grey dots
            ax1.scatter(
                X_space.flatten(), T_grid.flatten(), c="lightgrey", s=15, alpha=0.5
            )

            # Highlight computed points (all previous time levels)
            for prev_t in range(n + 1):
                ax1.scatter(x, t[prev_t] * np.ones_like(x), c="blue", s=25, alpha=0.7)

            # Highlight current time level points that are already computed
            for computed_i in range(1, i):
                ax1.scatter(x[computed_i], t[n + 1], c="green", s=40, marker="s")

            # Highlight boundary points at current time level
            ax1.scatter(x[0], t[n + 1], c="red", s=50, marker="s", label="Boundary")
            ax1.scatter(x[-1], t[n + 1], c="red", s=50, marker="s")

            # Highlight current stencil points
            stencil_x = [x[i - 1], x[i], x[i + 1]]
            stencil_t = [t[n], t[n], t[n]]
            ax1.scatter(
                stencil_x,
                stencil_t,
                c="orange",
                s=80,
                marker="^",
                edgecolor="darkorange",
                linewidth=2,
                label="Stencil points",
            )

            # Highlight target point being computed
            ax1.scatter(
                x[i],
                t[n + 1],
                c="lime",
                s=100,
                marker="s",
                edgecolor="darkgreen",
                linewidth=3,
                label=f"Computing u[{i},{n+1}]",
            )

            # Draw stencil connections
            for sx, st in zip(stencil_x, stencil_t):
                ax1.plot([sx, x[i]], [st, t[n + 1]], "orange", linewidth=3, alpha=0.8)

            # Compute the new value using explicit scheme
            old_value = u[i, n + 1]  # This should be 0 initially
            u[i, n + 1] = u[i, n] + r * (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n])

            ax1.set_xlabel("Space (x)")
            ax1.set_ylabel("Time (t)")
            ax1.set_title(
                f"Heat Equation Explicit Stencil\nTime step {n+1}/{nt}, Point {i}/{nx-2}"
            )
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-0.05, L + 0.05)
            ax1.set_ylim(-0.01, T + 0.01)

            # Right plot: Current solution profile and stencil details
            # Plot current and previous solution profiles
            if n > 0:
                ax2.plot(
                    x,
                    u[:, n],
                    "b--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"t = {t[n]:.3f} (previous)",
                )
            ax2.plot(
                x, u[:, n + 1], "r-", linewidth=2, label=f"t = {t[n+1]:.3f} (current)"
            )
            ax2.plot(x, u[:, 0], "k:", alpha=0.5, label="Initial condition")

            # Highlight stencil points on solution plot
            ax2.scatter(
                [x[i - 1], x[i], x[i + 1]],
                [u[i - 1, n], u[i, n], u[i + 1, n]],
                c="orange",
                s=80,
                marker="^",
                edgecolor="darkorange",
                linewidth=2,
                zorder=5,
            )
            ax2.scatter(
                x[i],
                u[i, n + 1],
                c="lime",
                s=100,
                marker="s",
                edgecolor="darkgreen",
                linewidth=3,
                zorder=5,
            )

            ax2.set_xlabel("x")
            ax2.set_ylabel("u(x,t)")
            ax2.set_title("Solution Profile with Stencil")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add text box with stencil calculation details
            stencil_info = f"""Explicit Finite Difference Update:

Time step: n = {n} → n+1 = {n+1}
Spatial point: i = {i} (x = {x[i]:.3f})

Stencil values at time n = {n}:
  u[{i-1},{n}] = {u[i-1, n]:.6f}  (left)
  u[{i},{n}] = {u[i, n]:.6f}  (center)  
  u[{i+1},{n}] = {u[i+1, n]:.6f}  (right)

Parameters:
  r = α·dt/dx² = {r:.4f}
  dx = {dx:.4f}, dt = {dt:.6f}

Explicit scheme:
u[{i},{n+1}] = u[{i},{n}] + r·(u[{i+1},{n}] - 2·u[{i},{n}] + u[{i-1},{n}])
u[{i},{n+1}] = {u[i, n]:.6f} + {r:.4f}·({u[i+1, n]:.6f} - 2·{u[i, n]:.6f} + {u[i-1, n]:.6f})
u[{i},{n+1}] = {u[i, n]:.6f} + {r:.4f}·{u[i+1, n] - 2*u[i, n] + u[i-1, n]:.6f}
u[{i},{n+1}] = {u[i, n + 1]:.6f}

Progress: Point {i-1}/{nx-2} in time step {n+1}/{nt}
Stability: {"STABLE" if r <= 0.5 else "UNSTABLE"}"""

            ax3.text(
                0.05,
                0.95,
                stencil_info,
                transform=ax3.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
            )

            plt.tight_layout()
            frame_filename = f"temp_heat_stencil_frame_{len(frames):04d}.png"
            plt.savefig(frame_filename, dpi=100, bbox_inches="tight")
            frames.append(Image.open(frame_filename))
            plt.close()

    # Create GIF
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=300,  # 300ms per frame
        loop=0,
    )

    # Clean up temporary files
    for i in range(len(frames)):
        os.remove(f"temp_heat_stencil_frame_{i:04d}.png")

    print(f"Heat equation stencil animation saved as {output_filename}")
    print(f"Animation shows {len(frames)} frames")
    print(f"Progression: LEFT→RIGHT (space), then advance in TIME")
    print(
        f"Each interior point computed using 3-point stencil from previous time level"
    )
    print(f"Stability parameter r = {r:.4f} ({'STABLE' if r <= 0.5 else 'UNSTABLE'})")


def create_grid_update_animation(
    nx: int = 21, ny: int = 21, output_filename: str = "grid_stencil_animation.gif"
) -> None:
    """Creates enhanced animated GIF showing 2D Poisson equation stencil with solution evolution.

    Enhanced features:
    1. Visited points turn blue (from grey)
    2. Coarser grid for better visibility (max 13×13)
    3. Left-to-right stencil movement (proper Gauss-Seidel order)
    4. Solution appears with color-coded values (mountain/hill shape)
    5. Clear visualization of the iterative process

    Args:
        nx, ny: Grid dimensions (will be limited for better visualization)
        output_filename: Name of output GIF file
    """
    # Use coarser grid for better stencil visibility
    nx_vis = min(nx, 13)  # Limit to 13 points max for clear visualization
    ny_vis = min(ny, 13)  # Limit to 13 points max for clear visualization

    # Create coarser grid
    x = np.linspace(0, 1, nx_vis)
    y = np.linspace(0, 1, ny_vis)
    X, Y = np.meshgrid(x, y)

    # Define a test Poisson problem: ∇²u = -2π²sin(πx)sin(πy)
    # Analytical solution: u = sin(πx)sin(πy) (creates mountain/hill shape)
    def source_function(X, Y):
        return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    def boundary_zero(X, Y):
        return np.zeros_like(X)

    # Initialize solution
    u = np.zeros((ny_vis, nx_vis))
    dx = 1.0 / (nx_vis - 1)
    source = source_function(X, Y)

    # Apply boundary conditions (u = 0 on boundaries)
    u[0, :] = boundary_zero(X[0, :], Y[0, :])
    u[-1, :] = boundary_zero(X[-1, :], Y[-1, :])
    u[:, 0] = boundary_zero(X[:, 0], Y[:, 0])
    u[:, -1] = boundary_zero(X[:, -1], Y[:, -1])

    # Get all interior points in order (row by row, left to right)
    # CRITICAL: This creates the correct left-to-right, top-to-bottom order
    interior_points = []
    for i in range(1, ny_vis - 1):  # i = row index (y-coordinate, top to bottom)
        for j in range(1, nx_vis - 1):  # j = column index (x-coordinate, left to right)
            interior_points.append((i, j))

    print(f"Enhanced stencil animation - Grid: {nx_vis}×{ny_vis}")
    print(f"Interior points order (first 10): {interior_points[:10]}")
    print(f"Total interior points: {len(interior_points)}")

    # Use more sweeps to show convergence process
    n_sweeps = 20  # Show 20 sweeps to demonstrate convergence

    # Calculate global min/max for consistent colorbar using analytical solution
    analytical_solution = np.sin(np.pi * X) * np.sin(np.pi * Y)
    vmin, vmax = np.min(analytical_solution), np.max(analytical_solution)
    if vmax - vmin < 1e-10:
        vmin, vmax = -0.1, 0.1

    frames: list[Image.Image] = []

    # Track visited points for blue coloring
    visited_points = set()

    # Show selective point updates to keep animation manageable
    for sweep in range(n_sweeps):
        for point_idx, (i, j) in enumerate(interior_points):
            # Skip frames strategically to keep file size reasonable while showing convergence
            if sweep == 0:
                # Show every point in first sweep for complete understanding
                skip_frame = False
            elif sweep < 5:
                # Show every 2nd point in sweeps 2-5
                skip_frame = point_idx % 2 != 0
            elif sweep < 10:
                # Show every 4th point in sweeps 6-10
                skip_frame = point_idx % 4 != 0
            else:
                # Show every 8th point in later sweeps (11-20)
                skip_frame = point_idx % 8 != 0

            if skip_frame:
                # Still update the solution, just don't create a frame
                u[i, j] = 0.25 * (
                    u[i + 1, j]
                    + u[i - 1, j]
                    + u[i, j + 1]
                    + u[i, j - 1]
                    - dx**2 * source[i, j]
                )
                visited_points.add((i, j))
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

            # Left plot: Enhanced solution visualization with visited points
            # Create background solution contour (mountain/hill shape appearing)
            levels = np.linspace(vmin, vmax, 21)
            im = ax1.contourf(
                X, Y, u, levels=levels, cmap="viridis", alpha=0.7, vmin=vmin, vmax=vmax
            )
            ax1.contour(
                X, Y, u, levels=levels, colors="white", alpha=0.4, linewidths=0.8
            )

            # Plot ALL grid points with different colors based on status
            for gi in range(ny_vis):
                for gj in range(nx_vis):
                    # Determine point color and size based on status
                    if gi == 0 or gi == ny_vis - 1 or gj == 0 or gj == nx_vis - 1:
                        # Boundary points - red squares
                        ax1.scatter(
                            X[gi, gj],
                            Y[gi, gj],
                            c="red",
                            s=80,
                            marker="s",
                            edgecolor="darkred",
                            linewidth=2,
                            alpha=0.9,
                            zorder=4,
                        )
                    elif (gi, gj) in visited_points:
                        # Visited interior points - blue circles
                        ax1.scatter(
                            X[gi, gj],
                            Y[gi, gj],
                            c="blue",
                            s=60,
                            marker="o",
                            edgecolor="darkblue",
                            linewidth=1.5,
                            alpha=0.8,
                            zorder=3,
                        )
                    elif gi == i and gj == j:
                        # Current point being updated - large yellow square
                        ax1.scatter(
                            X[gi, gj],
                            Y[gi, gj],
                            c="yellow",
                            s=150,
                            marker="s",
                            edgecolor="orange",
                            linewidth=3,
                            alpha=1.0,
                            zorder=6,
                        )
                    elif (gi, gj) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        # Stencil neighbor points - lime triangles
                        ax1.scatter(
                            X[gi, gj],
                            Y[gi, gj],
                            c="lime",
                            s=100,
                            marker="^",
                            edgecolor="darkgreen",
                            linewidth=2,
                            alpha=0.9,
                            zorder=5,
                        )
                    else:
                        # Unvisited interior points - grey circles
                        ax1.scatter(
                            X[gi, gj],
                            Y[gi, gj],
                            c="lightgrey",
                            s=50,
                            marker="o",
                            edgecolor="grey",
                            linewidth=1,
                            alpha=0.6,
                            zorder=2,
                        )

            # Draw stencil connections with thicker, more visible lines
            stencil_connections = [
                ([X[i - 1, j], X[i, j]], [Y[i - 1, j], Y[i, j]]),  # North
                ([X[i + 1, j], X[i, j]], [Y[i + 1, j], Y[i, j]]),  # South
                ([X[i, j - 1], X[i, j]], [Y[i, j - 1], Y[i, j]]),  # West
                ([X[i, j + 1], X[i, j]], [Y[i, j + 1], Y[i, j]]),  # East
            ]

            for conn_x, conn_y in stencil_connections:
                ax1.plot(conn_x, conn_y, "lime", linewidth=5, alpha=0.8, zorder=4)

            # Update the current point using Gauss-Seidel iteration
            old_value = u[i, j]
            u[i, j] = 0.25 * (
                u[i + 1, j]
                + u[i - 1, j]
                + u[i, j + 1]
                + u[i, j - 1]
                - dx**2 * source[i, j]
            )

            # Mark this point as visited
            visited_points.add((i, j))

            ax1.set_xlabel("x", fontsize=14)
            ax1.set_ylabel("y", fontsize=14)
            ax1.set_title(
                f"Enhanced 2D Poisson Stencil Animation\n"
                f"Sweep {sweep+1}/{n_sweeps}, Point {point_idx+1}/{len(interior_points)} "
                f"(Grid: {nx_vis}×{ny_vis})",
                fontsize=16,
            )
            ax1.set_aspect("equal")

            # Enhanced colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
            cbar.set_label("Solution u(x,y)", fontsize=12)

            # Add comprehensive legend
            legend_elements = [
                plt.scatter(
                    [],
                    [],
                    c="lightgrey",
                    s=50,
                    marker="o",
                    edgecolor="grey",
                    label="Unvisited points",
                ),
                plt.scatter(
                    [],
                    [],
                    c="blue",
                    s=60,
                    marker="o",
                    edgecolor="darkblue",
                    label="Visited points",
                ),
                plt.scatter(
                    [],
                    [],
                    c="yellow",
                    s=150,
                    marker="s",
                    edgecolor="orange",
                    label="Current point",
                ),
                plt.scatter(
                    [],
                    [],
                    c="lime",
                    s=100,
                    marker="^",
                    edgecolor="darkgreen",
                    label="Stencil neighbors",
                ),
                plt.scatter(
                    [],
                    [],
                    c="red",
                    s=80,
                    marker="s",
                    edgecolor="darkred",
                    label="Boundary points",
                ),
            ]
            ax1.legend(
                handles=legend_elements,
                loc="upper left",
                bbox_to_anchor=(0, 1),
                fontsize=11,
                framealpha=0.9,
            )

            # Right plot: Enhanced information display
            ax2.axis("off")

            # Calculate progress statistics
            total_interior = len(interior_points)
            points_completed = len(visited_points)
            progress_percent = (points_completed / total_interior) * 100

            # Show detailed information
            info_text = f"""Enhanced 2D Poisson Equation Stencil Visualization

PROBLEM: ∇²u = -2π²sin(πx)sin(πy)
SOLUTION: u = sin(πx)sin(πy) (mountain/hill shape)

GRID INFORMATION:
• Grid size: {nx_vis} × {ny_vis} points
• Interior points: {total_interior}
• Boundary conditions: u = 0 on all edges

CURRENT STATUS:
• Sweep: {sweep+1}/{n_sweeps}
• Point: {point_idx+1}/{total_interior} in current sweep
• Progress: {progress_percent:.1f}% complete
• Grid position: row {i}, col {j}
• Physical coordinates: ({X[i,j]:.3f}, {Y[i,j]:.3f})

GAUSS-SEIDEL UPDATE:
Current value: u[{i},{j}] = {old_value:.6f}

Stencil neighbors (using latest values):
  North: u[{i-1},{j}] = {u[i-1, j]:.6f}
  South: u[{i+1},{j}] = {u[i+1, j]:.6f}
  West:  u[{i},{j-1}] = {u[i, j-1]:.6f}
  East:  u[{i},{j+1}] = {u[i, j+1]:.6f}

Source term: f[{i},{j}] = {source[i, j]:.6f}

Update formula:
u_new = (u_N + u_S + u_W + u_E - h²f) / 4
u_new = {u[i, j]:.6f}

Change: Δu = {u[i, j] - old_value:.6f}

SOLUTION EVOLUTION:
• Mountain/hill shape appearing at center
• Maximum at (0.5, 0.5) ≈ {np.max(analytical_solution):.3f}
• Analytical solution: {analytical_solution[i, j]:.6f}
• Current error: {abs(u[i, j] - analytical_solution[i, j]):.2e}

TRAVERSAL: LEFT→RIGHT, then TOP→BOTTOM
Row {i}: ({i},1) → ({i},2) → ... → ({i},{nx_vis-2})"""

            ax2.text(
                0.05,
                0.95,
                info_text,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcyan", alpha=0.95),
            )

            plt.tight_layout()
            frame_filename = f"temp_enhanced_stencil_frame_{len(frames):04d}.png"
            plt.savefig(frame_filename, dpi=120, bbox_inches="tight")
            frames.append(Image.open(frame_filename))
            plt.close()

    # Create GIF with faster speed for 20 sweeps
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=150,  # 150ms per frame for faster viewing of 20 sweeps
        loop=0,
    )

    # Clean up
    for i in range(len(frames)):
        os.remove(f"temp_enhanced_stencil_frame_{i:04d}.png")

    print(f"Enhanced 2D Poisson stencil animation saved as {output_filename}")
    print(f"Grid: {nx_vis}×{ny_vis} (coarser for better visibility)")
    print(f"Animation shows {len(frames)} frames over {n_sweeps} sweeps")
    print(f"Features: visited points turn blue, solution appears, left→right movement")
    print(f"Mountain/hill solution shape clearly visible as it evolves")
    print(
        f"Frame sampling: full sweep 1, every 2nd point sweeps 2-5, every 4th sweeps 6-10, every 8th sweeps 11-20"
    )
    print(f"Animation speed: 150ms per frame for faster viewing of convergence process")


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

    Creates multiple subplots showing solution evolution, energy evolution,
    temperature profiles, and various cross-sections.
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

    # 4. Energy evolution (L2 norm) - moved to position [1, 0]
    ax4 = axes[1, 0]
    energy = np.array([np.trapz(u[:, n] ** 2, x) for n in range(nt + 1)])
    ax4.plot(t, energy, "b-", linewidth=2)
    ax4.set_xlabel("Time (t)")
    ax4.set_ylabel("Energy (∫u²dx)")
    ax4.set_title("Energy Evolution")
    ax4.grid(True, alpha=0.3)

    # 5. Maximum temperature evolution - moved to position [1, 1]
    ax5 = axes[1, 1]
    max_temp = np.array([np.max(u[:, n]) for n in range(nt + 1)])
    min_temp = np.array([np.min(u[:, n]) for n in range(nt + 1)])
    ax5.plot(t, max_temp, "r-", linewidth=2, label="Maximum")
    ax5.plot(t, min_temp, "b-", linewidth=2, label="Minimum")
    ax5.set_xlabel("Time (t)")
    ax5.set_ylabel("Temperature")
    ax5.set_title("Temperature Extremes")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Remove the unused subplot [1, 2]
    axes[1, 2].remove()

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

    # Create stencil animations
    create_grid_update_animation(21, 21, "stencil_animation.gif")

    # Create heat equation stencil animation (left-to-right, then time advance)
    create_heat_stencil_animation(
        alpha=0.01,
        L=1.0,
        T=1.0,  # Changed from 0.1 to 1.0
        nx=21,
        nt=10,
        initial_condition=initial_gaussian,
        output_filename="heat_stencil_march.gif",
    )

    # Plot comprehensive results
    plot_poisson_results(source_function, boundary_zero, Lx, Ly, nx, ny)

    print("\nAll PDE demonstrations completed!")
    print("Generated files:")
    print("- heat_evolution.gif")
    print("- heat_lattice.gif")
    print("- poisson_jacobi.gif")
    print("- poisson_gauss_seidel.gif")
    print("- stencil_animation.gif")
    print("- heat_stencil_march.gif")
    print("- heat_equation_analysis.png")
    print("- poisson_equation_analysis.png")
