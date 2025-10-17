"""Finite Difference Methods for solving ODEs.

This module implements various numerical methods (Forward Euler, Backward Euler,
Trapezoidal, and RK4) to solve the simple ODE: dy/dt = lambda * y.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from PIL import Image
import os
from pathlib import Path


def forward_euler(
    lambda_val: float, y0: float, t_span: tuple[float, float], h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solves ODE using Forward Euler method.

    Implements the explicit forward euler scheme:
    y_{k+1} = y_k + h*lambda*y_k.

    Args:
        lambda_val: The parameter lambda in dy/dt = lambda * y
        y0: The initial condition.
        t_span: Tuple of t_start and t_end.
        h: Time step size.

    Returns:
        Tuple of (time_points, numerical_solution, global_errors, truncation_errors).
    """
    t_start, t_end = t_span
    n_steps: int = int((t_end - t_start) / h)
    t: np.ndarray = np.linspace(t_start, t_end, n_steps + 1)
    y: np.ndarray = np.zeros(n_steps + 1)
    y[0] = y0

    global_errors: np.ndarray = np.zeros(n_steps + 1)
    truncation_errors: np.ndarray = np.zeros(n_steps)

    for k in range(n_steps):
        # Forward Euler step:
        y[k + 1] = y[k] + h * lambda_val * y[k]

        # Compute true solution at t[k+1]
        y_true: float = y0 * np.exp(lambda_val * t[k + 1])
        global_errors[k + 1] = y_true - y[k + 1]

        # Compute truncation error
        # tau_k = (y(t_{k+1}) - y(t_k))/h - Phi(t_k, y(t_k); h)
        # For Forward Euler: Phi = f(t_k, y(t_k)) = lambda * y(t_k)
        y_true_k: float = y0 * np.exp(lambda_val * t[k])
        y_true_k_plus_1: float = y0 * np.exp(lambda_val * t[k + 1])
        phi: float = lambda_val * y_true_k
        truncation_errors[k] = (y_true_k_plus_1 - y_true_k) / h - phi

    return t, y, global_errors, truncation_errors


def backward_euler(
    lambda_val: float, y0: float, t_span: tuple[float, float], h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solves ODE using Backward Euler method.

    Implements the implicit backward euler scheme:
    y_{k+1} = y_k + h*lambda*y_{k+1}.

    Args:
        lambda_val: The parameter lambda in dy/dt = lambda * y
        y0: The initial condition.
        t_span: Tuple of t_start and t_end.
        h: Time step size.

    Returns:
        Tuple of (time_points, numerical_solution, global_errors, truncation_errors).
    """
    t_start, t_end = t_span
    n_steps: int = int((t_end - t_start) / h)

    t: np.ndarray = np.linspace(t_start, t_end, n_steps + 1)
    y: np.ndarray = np.zeros(n_steps + 1)
    y[0] = y0

    global_errors: np.ndarray = np.zeros(n_steps + 1)
    truncation_errors: np.ndarray = np.zeros(n_steps)

    for k in range(n_steps):
        # Backward Euler step: y_{k+1} = y_k + h*lambda*y_{k+1}
        # Rearrange to solve for y_{k+1}
        # y_{k+1} * (1 - h*lambda) = y_k
        y[k + 1] = y[k] / (1 - h * lambda_val)

        # Compute true solution at t[k+1]
        y_true: float = y0 * np.exp(lambda_val * t[k + 1])
        global_errors[k + 1] = y_true - y[k + 1]

        # Compute truncation error
        # For backward Euler: Phi = f(t_{k+1}, y(t_{k+1})) = lambda * y(t_{k+1})
        y_true_k: float = y0 * np.exp(lambda_val * t[k])
        y_true_k_plus_1: float = y0 * np.exp(lambda_val * t[k + 1])
        phi: float = lambda_val * y_true_k_plus_1
        truncation_errors[k] = (y_true_k_plus_1 - y_true_k) / h - phi

    return t, y, global_errors, truncation_errors


def trapezoidal_method(
    lambda_val: float, y0: float, t_span: tuple[float, float], h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solves ODE using Trapezoidal method.

    Implements the implicit trapezoidal scheme:
    y_{k+1} = y_k + h/2*(f(t_k, y_k) + f(t_{k+1}, y_{k+1})).

    Args:
        lambda_val: The parameter lambda in dy/dt = lambda * y
        y0: The initial condition.
        t_span: Tuple of t_start and t_end.
        h: Time step size.

    Returns:
        Tuple of (time_points, numerical_solution, global_errors, truncation_errors).
    """
    t_start, t_end = t_span
    n_steps: int = int((t_end - t_start) / h)

    t: np.ndarray = np.linspace(t_start, t_end, n_steps + 1)
    y: np.ndarray = np.zeros(n_steps + 1)
    y[0] = y0

    global_errors: np.ndarray = np.zeros(n_steps + 1)
    truncation_errors: np.ndarray = np.zeros(n_steps)

    for k in range(n_steps):
        # Trapezoidal step:
        # y_{k+1} = y_k + h/2*(lambda*y_k + lambda*y_{k+1})
        # Rearrange to solve for y_{k+1}
        # y_{k+1} * (1 - h*lambda/2) = y_k * (1 + h*lambda/2)
        y[k + 1] = y[k] * (1 + h * lambda_val / 2) / (1 - h * lambda_val / 2)

        # Compute true solution at t[k+1]
        y_true: float = y0 * np.exp(lambda_val * t[k + 1])
        global_errors[k + 1] = y_true - y[k + 1]

        # Compute truncation error
        # For Trapezoidal: Phi = (f(t_k, y(t_k)) + f(t_{k+1}, y(t_{k+1})))/2
        y_true_k: float = y0 * np.exp(lambda_val * t[k])
        y_true_k_plus_1: float = y0 * np.exp(lambda_val * t[k + 1])
        phi: float = (lambda_val * y_true_k + lambda_val * y_true_k_plus_1) / 2
        truncation_errors[k] = (y_true_k_plus_1 - y_true_k) / h - phi

    return t, y, global_errors, truncation_errors


def rk4_method(
    lambda_val: float, y0: float, t_span: tuple[float, float], h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solves ODE using Runge-Kutta 4th order method.

    Implements the 4th order Runge-Kutta method:
    y_{k+1} = y_k + h/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
    where:
    k_1 = f(t_k, y_k)
    k_2 = f(t_k + h/2, y_k + h*k_1/2)
    k_3 = f(t_k + h/2, y_k + h*k_2/2)
    k_4 = f(t_k + h, y_k + h*k_3)

    Args:
        lambda_val: The parameter lambda in dy/dt = lambda * y
        y0: The initial condition.
        t_span: Tuple of t_start and t_end.
        h: Time step size.

    Returns:
        Tuple of (time_points, numerical_solution, global_errors, truncation_errors).
    """
    t_start, t_end = t_span
    n_steps: int = int((t_end - t_start) / h)

    t: np.ndarray = np.linspace(t_start, t_end, n_steps + 1)
    y: np.ndarray = np.zeros(n_steps + 1)
    y[0] = y0

    global_errors: np.ndarray = np.zeros(n_steps + 1)
    truncation_errors: np.ndarray = np.zeros(n_steps)

    def f(t_val: float, y_val: float) -> float:
        """ODE function: dy/dt = lambda * y"""
        return lambda_val * y_val

    for k in range(n_steps):
        # RK4 step
        k1 = f(t[k], y[k])
        k2 = f(t[k] + h / 2, y[k] + h * k1 / 2)
        k3 = f(t[k] + h / 2, y[k] + h * k2 / 2)
        k4 = f(t[k] + h, y[k] + h * k3)

        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Compute true solution at t[k+1]
        y_true: float = y0 * np.exp(lambda_val * t[k + 1])
        global_errors[k + 1] = y_true - y[k + 1]

        # Compute truncation error
        y_true_k: float = y0 * np.exp(lambda_val * t[k])
        y_true_k_plus_1: float = y0 * np.exp(lambda_val * t[k + 1])

        # For RK4, the method function Phi is more complex
        phi = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        truncation_errors[k] = (y_true_k_plus_1 - y_true_k) / h - phi

    return t, y, global_errors, truncation_errors


def true_solution(lambda_val: float, y0: float, t: np.ndarray) -> np.ndarray:
    """Computes true solution of the ODE.

    For the ODE dy/dt = lambda * y with initial condition y(0) = y0,
    the analytical solution is y(t) = y0 * exp(lambda * t).

    Args:
        lambda_val: The parameter lambda in dy/dt = lambda * y
        y0: The initial condition.
        t: Time points.

    Returns:
        True solution of the ODE at time points t.
    """
    return y0 * np.exp(lambda_val * t)


def plot_stability_regions() -> None:
    """Plots the stability regions for different numerical methods.

    Creates a plot showing the stability regions in the complex plane
    for Forward Euler, Backward Euler, Trapezoidal, and Runge-Kutta 4th order methods.
    """
    # Create complex plane grid
    x = np.linspace(-4, 2, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Stability functions for each method
    # Forward Euler: |1 + z| <= 1
    stability_fe = np.abs(1 + Z) <= 1

    # Backward Euler: |1/(1-z)| <= 1 => |1-z| >= 1
    stability_be = np.abs(1 - Z) >= 1

    # Trapezoidal: |(1 + z/2)/(1 - z/2)| <= 1
    stability_trap = np.abs((1 + Z / 2) / (1 - Z / 2)) <= 1

    # RK4: |1 + z + z^2/2 + z^3/6 + z^4/24| <= 1
    stability_rk4 = np.abs(1 + Z + Z**2 / 2 + Z**3 / 6 + Z**4 / 24) <= 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Stability Regions for Numerical Methods", fontsize=16)

    methods = [
        ("Forward Euler", stability_fe, "red"),
        ("Backward Euler", stability_be, "blue"),
        ("Trapezoidal", stability_trap, "green"),
        ("RK4", stability_rk4, "purple"),
    ]

    for i, (name, stability, color) in enumerate(methods):
        ax = axes[i // 2, i % 2]
        ax.contourf(
            X, Y, stability.astype(int), levels=[0.5, 1.5], colors=[color], alpha=0.3
        )
        ax.contour(
            X, Y, stability.astype(int), levels=[0.5], colors=[color], linewidths=2
        )
        ax.set_xlabel("Real(λh)")
        ax.set_ylabel("Imag(λh)")
        ax.set_title(f"{name} Stability Region")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("stability_regions.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_solutions_and_errors(
    lambda_val: float, y0: float, t_span: tuple[float, float], h: float
) -> None:
    """Plots numerical solutions, errors and comparison.

    Creates multiple subplots showing the true vs numerical solutions,
    global errors and truncation errors for all methods.

    Args:
        lambda_val: The parameter lambda in dy/dt = lambda * y
        y0: The initial condition.
        t_span: Tuple of t_start and t_end.
        h: Time step size.
    """
    # Compute solutions for all methods
    t_fe, y_fe, ge_fe, te_fe = forward_euler(lambda_val, y0, t_span, h)
    t_be, y_be, ge_be, te_be = backward_euler(lambda_val, y0, t_span, h)
    t_trap, y_trap, ge_trap, te_trap = trapezoidal_method(lambda_val, y0, t_span, h)
    t_rk4, y_rk4, ge_rk4, te_rk4 = rk4_method(lambda_val, y0, t_span, h)

    # True solution
    y_true = true_solution(lambda_val, y0, t_fe)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Numerical Solutions (λ={lambda_val}, h={h})", fontsize=16)

    # Solutions comparison
    ax1 = axes[0, 0]
    ax1.plot(t_fe, y_true, "k-", linewidth=2, label="True Solution")
    ax1.plot(t_fe, y_fe, "r--", label="Forward Euler")
    ax1.plot(t_be, y_be, "b--", label="Backward Euler")
    ax1.plot(t_trap, y_trap, "g--", label="Trapezoidal")
    ax1.plot(t_rk4, y_rk4, "m--", label="RK4")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("y(t)")
    ax1.set_title("Solutions Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Global errors
    ax2 = axes[0, 1]
    ax2.semilogy(t_fe, np.abs(ge_fe), "r-", label="Forward Euler")
    ax2.semilogy(t_be, np.abs(ge_be), "b-", label="Backward Euler")
    ax2.semilogy(t_trap, np.abs(ge_trap), "g-", label="Trapezoidal")
    ax2.semilogy(t_rk4, np.abs(ge_rk4), "m-", label="RK4")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("|Global Error|")
    ax2.set_title("Global Errors")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Truncation errors
    ax3 = axes[1, 0]
    ax3.semilogy(t_fe[:-1], np.abs(te_fe), "r-", label="Forward Euler")
    ax3.semilogy(t_be[:-1], np.abs(te_be), "b-", label="Backward Euler")
    ax3.semilogy(t_trap[:-1], np.abs(te_trap), "g-", label="Trapezoidal")
    ax3.semilogy(t_rk4[:-1], np.abs(te_rk4), "m-", label="RK4")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("|Truncation Error|")
    ax3.set_title("Truncation Errors")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Error vs step size (if we want to show convergence)
    ax4 = axes[1, 1]
    h_values = [0.1, 0.05, 0.025, 0.0125]
    final_errors = {"FE": [], "BE": [], "Trap": [], "RK4": []}

    for h_test in h_values:
        _, _, ge_fe_test, _ = forward_euler(lambda_val, y0, t_span, h_test)
        _, _, ge_be_test, _ = backward_euler(lambda_val, y0, t_span, h_test)
        _, _, ge_trap_test, _ = trapezoidal_method(lambda_val, y0, t_span, h_test)
        _, _, ge_rk4_test, _ = rk4_method(lambda_val, y0, t_span, h_test)

        final_errors["FE"].append(abs(ge_fe_test[-1]))
        final_errors["BE"].append(abs(ge_be_test[-1]))
        final_errors["Trap"].append(abs(ge_trap_test[-1]))
        final_errors["RK4"].append(abs(ge_rk4_test[-1]))

    ax4.loglog(h_values, final_errors["FE"], "r-o", label="Forward Euler")
    ax4.loglog(h_values, final_errors["BE"], "b-o", label="Backward Euler")
    ax4.loglog(h_values, final_errors["Trap"], "g-o", label="Trapezoidal")
    ax4.loglog(h_values, final_errors["RK4"], "m-o", label="RK4")
    ax4.set_xlabel("Step Size h")
    ax4.set_ylabel("Final Global Error")
    ax4.set_title("Convergence Analysis")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def precompute_all_solutions(
    lambda_vals: list[float],
    h_values: list[float],
    y0: float,
    t_span: tuple[float, float],
) -> dict[
    str,
    dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    """Precomputes solutions for all combinations of lambda and h.

    Args:
        lambda_vals: List of lambda values to compute.
        h_values: List of step sizes to compute.
        y0: The initial condition.
        t_span: Tuple of t_start and t_end.

    Returns:
        Dictionary with method names as keys and nested dictionaries as values.
        Inner dictionaries have (lambda, h) tuples as keys and solution tuples as values.
    """
    methods = {
        "Forward Euler": forward_euler,
        "Backward Euler": backward_euler,
        "Trapezoidal": trapezoidal_method,
        "RK4": rk4_method,
    }

    results = {method: {} for method in methods}

    for lambda_val in lambda_vals:
        for h in h_values:
            for method_name, method_func in methods.items():
                try:
                    result = method_func(lambda_val, y0, t_span, h)
                    results[method_name][(lambda_val, h)] = result
                except:
                    # Handle potential numerical issues (e.g., division by zero)
                    results[method_name][(lambda_val, h)] = None

    return results


def plot_interactive_solutions(
    lambda_vals: list[float],
    h_values: list[float],
    y0: float,
    t_span: tuple[float, float],
) -> None:
    """Creates an interactive plot with sliders for lambda and h values.

    Args:
        lambda_vals: List of lambda values for the slider.
        h_values: List of step sizes for the slider.
        y0: The initial condition.
        t_span: Tuple of t_start and t_end.
    """
    # Precompute all solutions
    results = precompute_all_solutions(lambda_vals, h_values, y0, t_span)

    # Create the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(bottom=0.25)

    # Create sliders
    ax_lambda = plt.axes([0.2, 0.1, 0.5, 0.03])
    ax_h = plt.axes([0.2, 0.05, 0.5, 0.03])

    slider_lambda = Slider(
        ax_lambda,
        "λ",
        min(lambda_vals),
        max(lambda_vals),
        valinit=lambda_vals[len(lambda_vals) // 2],
        valfmt="%.2f",
    )
    slider_h = Slider(
        ax_h,
        "h",
        min(h_values),
        max(h_values),
        valinit=h_values[len(h_values) // 2],
        valfmt="%.4f",
    )

    def update_plot(val):
        """Update plot based on slider values."""
        lambda_val = slider_lambda.val
        h = slider_h.val

        # Find closest precomputed values
        lambda_closest = min(lambda_vals, key=lambda x: abs(x - lambda_val))
        h_closest = min(h_values, key=lambda x: abs(x - h))

        # Clear all axes
        for ax in axes.flat:
            ax.clear()

        # Get solutions
        methods = ["Forward Euler", "Backward Euler", "Trapezoidal", "RK4"]
        colors = ["red", "blue", "green", "purple"]

        solution_data = {}
        for method in methods:
            if results[method][(lambda_closest, h_closest)] is not None:
                solution_data[method] = results[method][(lambda_closest, h_closest)]

        if solution_data:
            # Get time points and true solution
            t = solution_data[list(solution_data.keys())[0]][0]
            y_true = true_solution(lambda_closest, y0, t)

            # Plot solutions
            ax1 = axes[0, 0]
            ax1.plot(t, y_true, "k-", linewidth=2, label="True Solution")
            for i, (method, color) in enumerate(zip(methods, colors)):
                if method in solution_data:
                    _, y, _, _ = solution_data[method]
                    ax1.plot(t, y, "--", color=color, label=method)
            ax1.set_xlabel("Time")
            ax1.set_ylabel("y(t)")
            ax1.set_title(f"Solutions (λ={lambda_closest:.2f}, h={h_closest:.4f})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot global errors
            ax2 = axes[0, 1]
            for i, (method, color) in enumerate(zip(methods, colors)):
                if method in solution_data:
                    t, _, ge, _ = solution_data[method]
                    ax2.semilogy(t, np.abs(ge), color=color, label=method)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("|Global Error|")
            ax2.set_title("Global Errors")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot truncation errors
            ax3 = axes[1, 0]
            for i, (method, color) in enumerate(zip(methods, colors)):
                if method in solution_data:
                    t, _, _, te = solution_data[method]
                    ax3.semilogy(t[:-1], np.abs(te), color=color, label=method)
            ax3.set_xlabel("Time")
            ax3.set_ylabel("|Truncation Error|")
            ax3.set_title("Truncation Errors")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot stability region point
            ax4 = axes[1, 1]
            z = lambda_closest * h_closest
            ax4.plot(z.real, z.imag, "ko", markersize=10, label=f"λh = {z:.3f}")

            # Add stability boundaries (simplified)
            theta = np.linspace(0, 2 * np.pi, 100)
            # Forward Euler stability boundary: |1 + z| = 1
            circle_x = -1 + np.cos(theta)
            circle_y = np.sin(theta)
            ax4.plot(circle_x, circle_y, "r-", label="Forward Euler")

            ax4.set_xlabel("Real(λh)")
            ax4.set_ylabel("Imag(λh)")
            ax4.set_title("Stability Analysis")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color="k", linewidth=0.5)
            ax4.axvline(x=0, color="k", linewidth=0.5)

        plt.draw()

    # Connect sliders to update function
    slider_lambda.on_changed(update_plot)
    slider_h.on_changed(update_plot)

    # Initial plot
    update_plot(None)

    plt.show()


def create_method_animation_gif(
    method_name: str,
    lambda_val: float,
    y0: float,
    t_span: tuple[float, float],
    h: float,
    max_steps: int,
    output_filename: str,
) -> None:
    """Creates an enhanced animated GIF showing the evolution of a numerical method with method comparison.

    Shows slope estimates, tangent lines, and step arrows for the current method,
    plus comparison arrows for all other methods (Forward Euler, Backward Euler, RK4, Trapezoidal).

    Args:
        method_name: Name of the method ('Forward Euler', 'Backward Euler', 'Trapezoidal', 'RK4')
        lambda_val: The parameter lambda in dy/dt = lambda * y
        y0: The initial condition.
        t_span: Tuple of t_start and t_end.
        h: Time step size.
        max_steps: Maximum number of steps to animate.
        output_filename: Name of the output GIF file.
    """
    methods = {
        "Forward Euler": forward_euler,
        "Backward Euler": backward_euler,
        "Trapezoidal": trapezoidal_method,
        "RK4": rk4_method,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}")

    # Get full solution
    t_full, y_full, _, _ = methods[method_name](lambda_val, y0, t_span, h)
    y_true_full = true_solution(lambda_val, y0, t_full)

    # Limit to max_steps
    n_frames = min(max_steps, len(t_full))

    def calculate_next_point(
        t_curr: float, y_curr: float, method: str
    ) -> tuple[float, float]:
        """Calculate the next point for a given method."""
        if method == "Forward Euler":
            y_next = y_curr + h * lambda_val * y_curr
        elif method == "Backward Euler":
            y_next = y_curr / (1 - h * lambda_val)
        elif method == "Trapezoidal":
            y_next = y_curr * (1 + h * lambda_val / 2) / (1 - h * lambda_val / 2)
        elif method == "RK4":
            k1 = lambda_val * y_curr
            k2 = lambda_val * (y_curr + h * k1 / 2)
            k3 = lambda_val * (y_curr + h * k2 / 2)
            k4 = lambda_val * (y_curr + h * k3)
            y_next = y_curr + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ValueError(f"Unknown method: {method}")

        return t_curr + h, y_next

    def calculate_slopes(t_curr: float, y_curr: float, method: str) -> dict:
        """Calculate slopes for visualization based on method."""
        slopes = {}

        if method == "Forward Euler":
            slopes["method"] = lambda_val * y_curr
            slopes["description"] = (
                f"f(t_n, y_n) = λy_n = {lambda_val:.3f} × {y_curr:.3f} = {slopes['method']:.3f}"
            )

        elif method == "Backward Euler":
            # For visualization, show the implicit slope that was used
            y_next = y_curr / (1 - h * lambda_val)
            slopes["method"] = lambda_val * y_next
            slopes["description"] = (
                f"f(t_{{n+1}}, y_{{n+1}}) = λy_{{n+1}} = {lambda_val:.3f} × {y_next:.3f} = {slopes['method']:.3f}"
            )

        elif method == "Trapezoidal":
            # Average of forward and backward slopes
            slope_forward = lambda_val * y_curr
            y_next = y_curr * (1 + h * lambda_val / 2) / (1 - h * lambda_val / 2)
            slope_backward = lambda_val * y_next
            slopes["method"] = (slope_forward + slope_backward) / 2
            slopes["description"] = (
                f"(f(t_n,y_n) + f(t_{{n+1}},y_{{n+1}}))/2 = ({slope_forward:.3f} + {slope_backward:.3f})/2 = {slopes['method']:.3f}"
            )

        elif method == "RK4":
            # Show all RK4 slopes
            k1 = lambda_val * y_curr
            k2 = lambda_val * (y_curr + h * k1 / 2)
            k3 = lambda_val * (y_curr + h * k2 / 2)
            k4 = lambda_val * (y_curr + h * k3)
            slopes["method"] = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            slopes["k1"] = k1
            slopes["k2"] = k2
            slopes["k3"] = k3
            slopes["k4"] = k4
            slopes["description"] = (
                f"(k1 + 2k2 + 2k3 + k4)/6 = ({k1:.3f} + 2×{k2:.3f} + 2×{k3:.3f} + {k4:.3f})/6 = {slopes['method']:.3f}"
            )

        return slopes

    # Create frames
    frames = []
    for i in range(1, n_frames + 1):
        fig = plt.figure(figsize=(12, 10))

        # Create subplot layout: plot on top, text box below
        ax = plt.subplot(2, 1, 1)  # Top plot
        ax_text = plt.subplot(2, 1, 2)  # Bottom text area
        ax_text.axis("off")  # Turn off axis for text area

        # Plot true solution
        t_plot = np.linspace(t_span[0], t_span[1], 1000)
        y_true_plot = true_solution(lambda_val, y0, t_plot)
        ax.plot(t_plot, y_true_plot, "k-", linewidth=2, label="True Solution")

        # Plot numerical solution up to current step
        ax.plot(
            t_full[: i + 1],
            y_full[: i + 1],
            "ro-",
            markersize=4,
            label=f"{method_name}",
        )

        # Current point
        t_curr = t_full[i - 1] if i > 0 else t_full[0]
        y_curr = y_full[i - 1] if i > 0 else y_full[0]

        # Next point (if not the first frame)
        if i < len(t_full):
            t_next = t_full[i]
            y_next = y_full[i]

            # Calculate slopes for current method
            slopes = calculate_slopes(t_curr, y_curr, method_name)

            # Draw slope line (tangent line) for current method
            line_length = h * 1.5  # Extend line a bit beyond the step
            t_line = np.array([t_curr - line_length / 2, t_curr + line_length / 2])
            y_line = y_curr + slopes["method"] * (t_line - t_curr)
            ax.plot(
                t_line, y_line, "g-", linewidth=3, alpha=0.8, label="Slope estimate"
            )

            # Draw arrows for ALL methods
            all_methods = ["Forward Euler", "Backward Euler", "RK4", "Trapezoidal"]
            method_colors = {
                "Forward Euler": "blue",
                "Backward Euler": "orange",
                "RK4": "purple",
                "Trapezoidal": "brown",
            }

            for method in all_methods:
                t_method_next, y_method_next = calculate_next_point(
                    t_curr, y_curr, method
                )

                if method == method_name:
                    # Current method: solid green arrow
                    ax.annotate(
                        "",
                        xy=(t_method_next, y_method_next),
                        xytext=(t_curr, y_curr),
                        arrowprops=dict(
                            arrowstyle="->", lw=3, color="green", alpha=1.0
                        ),
                    )
                else:
                    # Other methods: dashed colored arrows
                    ax.annotate(
                        "",
                        xy=(t_method_next, y_method_next),
                        xytext=(t_curr, y_curr),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=2,
                            color=method_colors[method],
                            alpha=0.6,
                            linestyle="dashed",
                        ),
                    )

            # For RK4, show intermediate slopes with reduced alpha
            if method_name == "RK4":
                # k1 slope (at beginning)
                t_k1 = np.array([t_curr, t_curr + h / 3])
                y_k1 = y_curr + slopes["k1"] * (t_k1 - t_curr)
                ax.plot(
                    t_k1,
                    y_k1,
                    "--",
                    color="orange",
                    alpha=0.6,
                    linewidth=2,
                    label="k1 slope",
                )

                # k2 slope (at midpoint)
                t_mid = t_curr + h / 2
                y_mid = y_curr + h * slopes["k1"] / 2
                t_k2 = np.array([t_mid - h / 6, t_mid + h / 6])
                y_k2 = y_mid + slopes["k2"] * (t_k2 - t_mid)
                ax.plot(
                    t_k2,
                    y_k2,
                    "--",
                    color="purple",
                    alpha=0.6,
                    linewidth=2,
                    label="k2 slope",
                )

                # Mark intermediate points
                ax.plot(t_mid, y_mid, "o", color="orange", markersize=6, alpha=0.7)

            # Add enhanced information box with larger timestamp and method comparison
            info_text = f"STEP {i}: t = {t_curr:.3f}, y = {y_curr:.3f}\n"
            info_text += f"h = {h:.3f}\n\n"
            info_text += f"CURRENT METHOD: {method_name}\n"
            info_text += f"{slopes['description']}\n"
            info_text += f"Next point: y_{{n+1}} = {y_next:.3f}\n\n"
            info_text += "METHOD COMPARISON:\n"

            # Show next points for all methods
            for method in all_methods:
                t_comp_next, y_comp_next = calculate_next_point(t_curr, y_curr, method)
                color_name = method_colors[method] if method != method_name else "GREEN"
                style = "SOLID" if method == method_name else "DASHED"
                info_text += f"• {method}: {y_comp_next:.3f} ({color_name}, {style})\n"

            # Position the text box in the bottom area
            ax_text.text(
                0.05,
                0.95,
                info_text,
                transform=ax_text.transAxes,
                fontsize=12,  # Increased for better readability
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcyan", alpha=0.9),
            )

        # Highlight current point (no green background, just a prominent marker)
        ax.plot(
            t_curr,
            y_curr,
            "ko",  # Black center
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=3,
            zorder=10,
        )
        # Add a red ring around it
        ax.plot(
            t_curr,
            y_curr,
            "ro",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=2,
            zorder=11,
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("y(t)")
        ax.set_title(
            f"{method_name} Evolution with Method Comparison (Step {i}/{n_frames-1})",
            fontsize=14,
        )
        # Create custom legend including method arrows
        legend_elements = ax.get_legend_handles_labels()[0]
        legend_labels = ax.get_legend_handles_labels()[1]

        # Add method arrow indicators to legend
        if i < len(t_full):
            from matplotlib.lines import Line2D

            legend_elements.extend(
                [
                    Line2D(
                        [0], [0], color="green", lw=3, label=f"{method_name} (current)"
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="blue",
                        lw=2,
                        linestyle="--",
                        alpha=0.6,
                        label="Forward Euler",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="orange",
                        lw=2,
                        linestyle="--",
                        alpha=0.6,
                        label="Backward Euler",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="purple",
                        lw=2,
                        linestyle="--",
                        alpha=0.6,
                        label="RK4",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="brown",
                        lw=2,
                        linestyle="--",
                        alpha=0.6,
                        label="Trapezoidal",
                    ),
                ]
            )

        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t_span)

        # Set y limits based on the solution behavior
        y_min = min(np.min(y_true_full), np.min(y_full)) * 1.1
        y_max = max(np.max(y_true_full), np.max(y_full)) * 1.1
        ax.set_ylim(y_min, y_max)

        # Save frame
        plt.tight_layout()
        plt.savefig(f"temp_frame_{i:03d}.png", dpi=100, bbox_inches="tight")
        frames.append(Image.open(f"temp_frame_{i:03d}.png"))
        plt.close()

    # Create GIF
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=400,  # Slower to see the details
        loop=0,
    )

    # Clean up temporary files
    for i in range(1, n_frames + 1):
        os.remove(f"temp_frame_{i:03d}.png")

    print(f"Enhanced animation saved as {output_filename}")


# Example usage and main execution
if __name__ == "__main__":
    # Example parameters
    lambda_val = -0.5
    y0 = 1.0
    t_span = (0.0, 5.0)
    h = 0.1

    # Plot stability regions
    plot_stability_regions()

    # Plot solutions and errors
    plot_solutions_and_errors(lambda_val, y0, t_span, h)

    # Create interactive plot
    lambda_vals = np.linspace(-2.0, 1.0, 21)
    h_values = np.logspace(-3, -1, 21)  # From 0.001 to 0.1
    plot_interactive_solutions(lambda_vals.tolist(), h_values.tolist(), y0, t_span)

    # Create animation GIFs for each method with larger time step for better slope visualization
    h_animation = 0.5  # Much larger time step to show slope differences clearly
    methods = ["Forward Euler", "Backward Euler", "Trapezoidal", "RK4"]
    for method in methods:
        filename = f"{method.lower().replace(' ', '_')}_animation.gif"
        create_method_animation_gif(
            method, lambda_val, y0, t_span, h_animation, 20, filename
        )
