import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Settings (change if needed)
# ----------------------------
N = 50
# grid dimensions: N x N
ITERATIONS = 5000  # number of relaxation iterations
V_HIGH = 1.0  # high potential boundary value
V_GROUND = 0.0  # grounded boundary value
# Defect settings (square block in the center)
DEFECT_HALF_WIDTH = 4  # defect will be (2*half_width + 1) by (2*half_width + 1)


# ----------------------------
# Helper: apply boundary conditions
# ----------------------------
def apply_boundary_conditions(V: np.ndarray) -> None:
    """
    Left boundary set to high potential.
    Other boundaries grounded.
    """
    V[:, 0] = V_HIGH  # left = high
    V[:, -1] = V_GROUND  # right = ground
    V[0, :] = V_GROUND  # top = ground
    V[-1, :] = V_GROUND  # bottom = ground


# ----------------------------
# Solver: finite difference relaxation (Gauss-Seidel style)
# ----------------------------
def solve_laplace(
    N: int, iterations: int, defect_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Solves Laplace's equation on an N x N grid using iterative neighbor-averaging.
    If defect_mask is provided, True cells are treated as "blocked" (not updated).
    """
    V = np.zeros((N, N), dtype=float)
    apply_boundary_conditions(V)
    for _ in range(iterations):
        # Update interior points
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if (defect_mask is not None) and defect_mask[i, j]:
                    continue  # blocked region (defect)
                V[i, j] = 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1])
            # Re-apply boundaries each iteration just to be safe
            apply_boundary_conditions(V)
    return V


# ----------------------------
# Make defect mask
# ----------------------------
def make_center_defect_mask(N: int, half_width: int) -> np.ndarray:
    """
    Creates a boolean mask with a square defect near the center.
    True = defect (blocked), False = normal update.
    """
    mask = np.zeros((N, N), dtype=bool)
    c = N // 2
    mask[c - half_width : c + half_width + 1, c - half_width : c + half_width + 1] = (
        True
    )
    return mask


# ----------------------------
# Plot helpers
# ----------------------------
def save_figure1_grid_setup(filename: str, N: int) -> None:
    grid = np.zeros((N, N))
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray")
    # plt.axis("off")
    # plt.title("2D Grid and Boundary Conditions", pad=20)

    # Labels (matches your paper figure)
    plt.text(N / 2, -3.5, "High Potential", ha="center", fontsize=11)
    plt.text(N / 2, N + 2.5, "Ground", ha="center", fontsize=11)
    plt.text(-4.5, N / 2, "Ground", rotation=90, va="center", fontsize=11)
    plt.text(N + 3.5, N / 2, "Ground", rotation=90, va="center", fontsize=11)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_heatmap(V: np.ndarray, filename: str, title: str, cmap: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(V, cmap=cmap)
    plt.colorbar(label="Electric Potential")
    # plt.title(title)
    # plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_difference_map(deltaV: np.ndarray, filename: str, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(deltaV, cmap="bwr")
    plt.colorbar(label="Change in Electric Potential")
    # plt.title(title)
    # plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Main run
# ----------------------------
def main():
    # Figure 1 (diagram)
    save_figure1_grid_setup("Figure1_GridSetup.png", N)

    # Baseline solution (no defect)
    V_base = solve_laplace(N, ITERATIONS)
    save_heatmap(
        V_base,
        "Figure2_BaselinePotential.png",
        "Steady-State Potential Distribution (No Defect)",
        cmap="inferno",
    )

    # Defect solution
    defect_mask = make_center_defect_mask(N, DEFECT_HALF_WIDTH)
    V_defect = solve_laplace(N, ITERATIONS, defect_mask=defect_mask)
    save_heatmap(
        V_defect,
        "Figure3_DefectPotential.png",
        "Potential Distribution with Internal Defect",
        cmap="inferno",
    )

    # Difference map (Figure 4)
    deltaV = V_defect - V_base
    save_difference_map(
        deltaV,
        "Figure4_PotentialDifference.png",
        "Difference in Electric Potential Due to an Internal Defect",
    )

    print("Done. Generated Figure1–Figure4 PNG files.")


if __name__ == "__main__":
    main()
