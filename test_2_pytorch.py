import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. GENERATE PARAMETERIZED TRAINING DATA
# ==========================================
n_samples = 20000
mu_b, sigma_b = 0.0, 1.0  # Fixed background

# Sample random signal parameters (mu_s, sigma_s) across a range for training
mu_s_train = np.random.uniform(1.0, 4.0, size=(n_samples, 1))
sigma_s_train = np.random.uniform(0.8, 2.0, size=(n_samples, 1))
theta_train = np.hstack([mu_s_train, sigma_s_train])  # Shape: (n_samples, 2)

# Generate data points
x_bkg = np.random.normal(loc=mu_b, scale=sigma_b, size=(n_samples, 1))
x_sig = np.random.normal(loc=mu_s_train, scale=sigma_s_train)

# THE FIX: Pair BOTH background and signal with the sampled theta parameters!
X_bkg_full = np.hstack([x_bkg, theta_train])  # [x, mu_s, sigma_s] for background
X_sig_full = np.hstack([x_sig, theta_train])  # [x, mu_s, sigma_s] for signal

Y_bkg = np.zeros((n_samples, 1))
Y_sig = np.ones((n_samples, 1))

X_train = np.vstack([X_bkg_full, X_sig_full])
Y_train = np.vstack([Y_bkg, Y_sig])

# Shuffle dataset
idx = np.random.permutation(len(X_train))
X_train, Y_train = X_train[idx], Y_train[idx]

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# ==========================================
# 2. DEFINE PARAMETERIZED NEURAL NETWORK
# ==========================================
class ParameterizedLRClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size is 3: [x, mu_s, sigma_s]
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = ParameterizedLRClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
epochs = 12
model.train()
print("Training the parameterized neural network...")
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
print("Training complete!\n")

# ==========================================
# 4. TEST WITH ANY CHOSEN FUNCTION PARAMETERS
# ==========================================
test_mu_s, test_sigma_s = 2.5, 1.5  # Custom test parameters
print(f"Testing network on custom signal: Normal(mu={test_mu_s}, sigma={test_sigma_s})")

x_grid = np.linspace(-3, 7, 500).reshape(-1, 1)
test_theta = np.full((500, 2), [test_mu_s, test_sigma_s])
x_grid_full = np.hstack([x_grid, test_theta])
x_grid_tensor = torch.tensor(x_grid_full, dtype=torch.float32)

model.eval()
with torch.no_grad():
    p_sig_given_x = model(x_grid_tensor).numpy()

p_sig_given_x = np.clip(p_sig_given_x, 1e-6, 1 - 1e-6)
learned_llr = np.log(p_sig_given_x / (1.0 - p_sig_given_x))

# True analytical LLR for this specific test configuration
pdf_bkg = (1.0 / (sigma_b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu_b) / sigma_b) ** 2)
pdf_sig = (1.0 / (test_sigma_s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - test_mu_s) / test_sigma_s) ** 2)
analytical_llr = np.log(pdf_sig / pdf_bkg)

# ==========================================
# 5. PLOTTING RESULTS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(x_grid, pdf_bkg, label='Background P(x|B)', color='blue', lw=2)
axes[0].plot(x_grid, pdf_sig, label=f'Custom Signal P(x|S, $\\mu$={test_mu_s})', color='orange', lw=2)
axes[0].set_title('Custom Signal vs Background Functions', fontsize=12)
axes[0].set_xlabel('x', fontsize=11)
axes[0].set_ylabel('Probability Density', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_grid, analytical_llr, label='Analytical LLR', color='black', linestyle='--', lw=2.5)
axes[1].plot(x_grid, learned_llr, label='Parameterized Net LLR', color='crimson', lw=2)
axes[1].set_title('Parameterized Likelihood Ratio Estimation', fontsize=12)
axes[1].set_xlabel('x', fontsize=11)
axes[1].set_ylabel('Log Likelihood Ratio t(x)', fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()