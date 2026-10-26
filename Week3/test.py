import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import ParameterizedLRClassifier

def test_model():
    # Load specific saved checkpoint from disk
    checkpoint_path = "checkpoints/model_epoch_15.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py first.")

    model = ParameterizedLRClassifier(input_dim=4)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print(f"Loaded weights from {checkpoint_path}")

    # ==========================================
    # 1. DEFINE TEST PARAMETERS & 4D GRID
    # ==========================================
    test_mu_s, test_sigma_s = 2.5, 1.5
    mu_b, sigma_b = 0.0, 1.0

    print(f"Testing 4D network on custom signal: Normal(mu={test_mu_s}, sigma={test_sigma_s})")

    n_points = 500
    x1_grid = np.linspace(-3, 7, n_points).reshape(-1, 1)
    x2_fixed = np.full((n_points, 1), 0.5)  # Fixing second dimension $x_2 = 0.5$ for slicing
    test_theta = np.full((n_points, 2), [test_mu_s, test_sigma_s])

    # Construct 4D test input: [x1, x2_fixed, mu_s, sigma_s]
    x_grid_full = np.hstack([x1_grid, x2_fixed, test_theta])
    x_grid_tensor = torch.tensor(x_grid_full, dtype=torch.float32)

    # ==========================================
    # 2. LEARNED VS ANALYTIC LLR
    # ==========================================
    with torch.no_grad():
        p_sig_given_x = model(x_grid_tensor).numpy()

    p_sig_given_x = np.clip(p_sig_given_x, 1e-6, 1 - 1e-6)
    learned_llr = np.log(p_sig_given_x / (1.0 - p_sig_given_x))

    # Analytical 2D Multivariate Gaussian PDFs & LLR
    pdf_bkg_1 = (1.0 / (sigma_b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1_grid - mu_b) / sigma_b) ** 2)
    pdf_bkg_2 = (1.0 / (sigma_b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x2_fixed - mu_b) / sigma_b) ** 2)
    pdf_bkg = pdf_bkg_1 * pdf_bkg_2

    pdf_sig_1 = (1.0 / (test_sigma_s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1_grid - test_mu_s) / test_sigma_s) ** 2)
    pdf_sig_2 = (1.0 / (test_sigma_s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x2_fixed - test_mu_s) / test_sigma_s) ** 2)
    pdf_sig = pdf_sig_1 * pdf_sig_2

    analytical_llr = np.log(pdf_sig / pdf_bkg)

    # ==========================================
    # 3. QUANTITATIVE APPROXIMATION CHECK
    # ==========================================
    learned_flat = learned_llr.squeeze()
    analytic_flat = analytical_llr.squeeze()

    mse = np.mean((learned_flat - analytic_flat) ** 2)
    mae = np.mean(np.abs(learned_flat - analytic_flat))
    correlation = np.corrcoef(learned_flat, analytic_flat)[0, 1]

    print("\n--- Quantitative LLR Approximation Check ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Pearson Correlation:       {correlation:.4f}")

    # ==========================================
    # 4. PLOTTING RESULTS
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x1_grid, pdf_bkg, label='Background P(x|B)', color='blue', lw=2)
    axes[0].plot(x1_grid, pdf_sig, label=f'Signal Slice P(x|S, $\\mu$={test_mu_s})', color='orange', lw=2)
    axes[0].set_title('4D Model: Observable Slice ($x_2 = 0.5$)', fontsize=12)
    axes[0].set_xlabel('$x_1$', fontsize=11)
    axes[0].set_ylabel('Probability Density', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x1_grid, analytic_flat, label='Analytical LLR', color='black', linestyle='--', lw=2.5)
    axes[1].plot(x1_grid, learned_flat, label='4D Parameterized Net LLR', color='crimson', lw=2)
    axes[1].set_title(f'LLR Comparison (Corr: {correlation:.3f}, MSE: {mse:.3f})', fontsize=12)
    axes[1].set_xlabel('$x_1$', fontsize=11)
    axes[1].set_ylabel('Log Likelihood Ratio $t(x)$', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("llr_4d_comparison.png", dpi=300)
    print("Saved plot to 'llr_4d_comparison.png'")
    plt.show()



    #heatmap 

    # Create a 2D grid for observables x1 and x2
    n_grid = 100
    x1_vals = np.linspace(-3, 7, n_grid)
    x2_vals = np.linspace(-3, 7, n_grid)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Flatten grid to pass into the 4D network
    grid_points = n_grid * n_grid
    test_theta_grid = np.full((grid_points, 2), [test_mu_s, test_sigma_s])
    x_grid_full = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1), test_theta_grid])
    x_grid_tensor = torch.tensor(x_grid_full, dtype=torch.float32)

    # Evaluate network over the 2D observable space
    with torch.no_grad():
        p_sig = model(x_grid_tensor).numpy().reshape(n_grid, n_grid)

    p_sig = np.clip(p_sig, 1e-6, 1 - 1e-6)
    learned_llr_2d = np.log(p_sig / (1.0 - p_sig))

    # Plotting the 2D contour map
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X1, X2, learned_llr_2d, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Learned LLR t(x1, x2)')
    plt.title(f'Learned LLR in 2D Observables ($x_1, x_2$)\n$\\mu_s={test_mu_s}, \\sigma_s={test_sigma_s}$', fontsize=12)
    plt.xlabel('$x_1$', fontsize=11)
    plt.ylabel('$x_2$', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("llr_4d_contour.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    test_model()