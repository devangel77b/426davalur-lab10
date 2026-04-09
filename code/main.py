import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle




# ----------------------------
# Settings
# ----------------------------
N = 50
ITERATIONS = 5000
V_HIGH = 1.0
V_GROUND = 0.0
# Defect settings
DEFECT_HALF_WIDTH = 4
# Physical size of square domain (meters)
Lx = 0.05
Ly = 0.05





# ----------------------------
# Helper: apply boundary conditions
# ----------------------------
def apply_boundary_conditions(V: np.ndarray) -> None:
    """
    Left boundary = high potential
    All other boundaries = grounded
    """
    V[:, 0] = V_HIGH
    V[:, -1] = V_GROUND
    V[0, :] = V_GROUND
    V[-1, :] = V_GROUND






    
# ----------------------------
# Solver: finite difference relaxation
# ----------------------------
def solve_laplace(N: int, iterations: int, defect_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Solves Laplace's equation on an N x N grid using iterative relaxation.
    If defect_mask is provided, True cells are treated as blocked and not updated.
    """
    V = np.zeros((N, N), dtype=float)
    apply_boundary_conditions(V)
    for _ in range(iterations):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if defect_mask is not None and defect_mask[i, j]:
                    continue
                V[i, j] = 0.25 * (
                    V[i + 1, j] +
                    V[i - 1, j] +
                    V[i, j + 1] +
                    V[i, j - 1]
                )
        apply_boundary_conditions(V)
        return V





    
# ----------------------------
# Make defect mask
# ----------------------------
def make_center_defect_mask(N: int, half_width: int) -> np.ndarray:
    """
    Creates a square defect mask in the center.
    True = defect region
    """
    mask = np.zeros((N, N), dtype=bool)
    c = N // 2
    mask[
        c - half_width:c + half_width + 1,
        c - half_width:c + half_width + 1
    ] = True
    return mask








# ----------------------------
# Compute electric field
# ----------------------------
def compute_electric_field(V: np.ndarray, Lx: float, Ly: float):
    """
    Computes E = -grad(V)
    """
    dx = Lx / (V.shape[1] - 1)
    dy = Ly / (V.shape[0] - 1)
    dVdy, dVdx = np.gradient(V, dy, dx)
    Ex = -dVdx
    Ey = -dVdy
    return Ex, Ey








# ----------------------------
# Figure 1: grid setup
# ----------------------------
def save_figure1_grid_setup(filename: str, N: int) -> None:
    grid = np.zeros((N, N))
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray", origin="lower")
    plt.text(N / 2, N + 2.5, "Ground", ha="center", fontsize=11)
    plt.text(N / 2, -4, "Ground", ha="center", fontsize=11)
    plt.text(-5, N / 2, "High Potential", rotation=90, va="center", fontsize=11)
    plt.text(N + 3.5, N / 2, "Ground", rotation=90, va="center", fontsize=11)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()





    
# ----------------------------
# Heatmap + vector field
# ----------------------------
def save_heatmap_with_field(
        V: np.ndarray,
        filename: str,
        title: str,
        cmap: str = "viridis",
        defect_mask: np.ndarray | None = None,
        annotate_defect: bool = False
) -> None:
    Ex, Ey = compute_electric_field(V, Lx, Ly)
    # coordinate grid in meters
    x = np.linspace(0, Lx, V.shape[1])
    y = np.linspace(0, Ly, V.shape[0])
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(6.2, 5.6))
    ax = plt.gca()
    im = ax.imshow(
        V,
        cmap=cmap,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="equal",
        vmin=0,
        vmax=1
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Electric Potential (V)")

    # Subsample arrows so plot isn't overcrowded
    step = 4
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    Exq = Ex[::step, ::step]
    Eyq = Ey[::step, ::step]
    
    # If defect exists, do not draw arrows inside it
    if defect_mask is not None:
        maskq = defect_mask[::step, ::step]
        Exq = np.where(maskq, np.nan, Exq)
        Eyq = np.where(maskq, np.nan, Eyq)
    ax.quiver(
        Xq, Yq, Exq, Eyq,
        color="black",
        scale=120,
        width=0.003,
        headwidth=3,
        headlength=4
    )
    
    # Draw defect box if present
    if defect_mask is not None:
        rows, cols = np.where(defect_mask)
        i_min, i_max = rows.min(), rows.max()
        j_min, j_max = cols.min(), cols.max()
        
        dx = Lx / (V.shape[1] - 1)
        dy = Ly / (V.shape[0] - 1)
        x0 = j_min * dx
        y0 = i_min * dy
        width = (j_max - j_min + 1) * dx
        height = (i_max - i_min + 1) * dy
        rect = Rectangle(
            (x0, y0), width, height,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
            linestyle="--"
        )
        ax.add_patch(rect)
        
        if annotate_defect:
            ax.text(
                x0 + width / 2,
                y0 + height / 2,
                "Defect Region",
                color="black",
                fontsize=8,
                ha="center",
                va="center"
            )
            
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()




    
# ----------------------------
# Difference map
# ----------------------------
def save_difference_map(deltaV: np.ndarray, filename: str, title: str) -> None:
    plt.figure(figsize=(6.2, 5.6))
    ax = plt.gca()
    im = ax.imshow(
        deltaV,
        cmap="bwr",
        origin="lower"
        extent=[0, Lx, 0, Ly],
        aspect="equal"
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Change in Electric Potential (V)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()





    
# ----------------------------
# Main
# ----------------------------
def main():
    # Figure 1
    save_figure1_grid_setup("Figure1_GridSetup.png", N)
    
    # No defect
    V_base = solve_laplace(N, ITERATIONS)
    save_heatmap_with_field(
        V_base,
        "Figure2_BaselinePotential.png",
        "Steady-State Potential Distribution (No Defect)",
        cmap="viridis"
    )
    
    # With defect
    defect_mask = make_center_defect_mask(N, DEFECT_HALF_WIDTH)
    V_defect = solve_laplace(N, ITERATIONS, defect_mask=defect_mask)
    save_heatmap_with_field(
        V_defect,
        "Figure3_DefectPotential.png",
        "Potential Distribution with Internal Defect",
        cmap="viridis",
        defect_mask=defect_mask,
        annotate_defect=True
    )

    # Difference map
    deltaV = V_defect - V_base
    save_difference_map(
        deltaV,
        "Figure4_PotentialDifference.png",
        "Difference in Electric Potential Due to an Internal Defect"
    )
    
    print("Done. Generated Figure1–Figure4 PNG files.")









if __name__ == "__main__":
    main()
