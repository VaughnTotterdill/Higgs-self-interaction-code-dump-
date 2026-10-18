import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. GENERATE TRAINING DATA
# ==========================================
n_samples = 10000
mu_b, sigma_b = 0.0, 1.0  # Background parameters
mu_s, sigma_s = 2.0, 1.2  # Signal parameters

x_bkg_train = np.random.normal(loc=mu_b, scale=sigma_b, size=(n_samples, 1))
y_bkg_train = np.zeros((n_samples, 1))

x_sig_train = np.random.normal(loc=mu_s, scale=sigma_s, size=(n_samples, 1))
y_sig_train = np.ones((n_samples, 1))

X_train = np.vstack([x_bkg_train, x_sig_train])
Y_train = np.vstack([y_bkg_train, y_sig_train])
idx = np.random.permutation(len(X_train))
X_train, Y_train = X_train[idx], Y_train[idx]

# Convert to PyTorch tensors and create DataLoader
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# ==========================================
# 2. DEFINE & TRAIN NEURAL NETWORK
# ==========================================
class LikelihoodRatioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Outputs probability P(y=1 | x)
        )
    
    def forward(self, x):
        return self.net(x)

model = LikelihoodRatioClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
model.train()
print("Training the neural network...")
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
print("Training complete!\n")

# ==========================================
# 3. TEST ON A BRAND-NEW, UNSEEN DATASET
# ==========================================
n_test = 5
print(f"--- Evaluating on a brand-new test set of {n_test * 2} random points ---")
x_test_bkg = np.random.normal(loc=mu_b, scale=sigma_b, size=(n_test, 1))
x_test_sig = np.random.normal(loc=mu_s, scale=sigma_s, size=(n_test, 1))
X_test_raw = np.vstack([x_test_bkg, x_test_sig])

# Convert test data to tensor
X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32)

# Run through trained model (no gradients needed during testing)
model.eval()
with torch.no_grad():
    p_test = model(X_test_tensor).numpy()

# Prevent division by zero, then compute likelihood ratio t(x)
p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
test_llrs = np.log(p_test / (1.0 - p_test))

for i in range(len(X_test_raw)):
    print(f"Data point x = {X_test_raw[i][0]:+.3f} | Computed LLR t(x) = {test_llrs[i][0]:+.3f}")
print("-" * 65 + "\n")

# ==========================================
# 4. GRID EVALUATION & PLOTTING
# ==========================================
x_grid = np.linspace(-3, 7, 500).reshape(-1, 1)
x_grid_tensor = torch.tensor(x_grid, dtype=torch.float32)

with torch.no_grad():
    p_sig_given_x = model(x_grid_tensor).numpy()

p_sig_given_x = np.clip(p_sig_given_x, 1e-6, 1 - 1e-6)
learned_llr = np.log(p_sig_given_x / (1.0 - p_sig_given_x))

# Analytical (True) PDFs and Log-Likelihood Ratio
pdf_bkg = (1.0 / (sigma_b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu_b) / sigma_b) ** 2)
pdf_sig = (1.0 / (sigma_s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu_s) / sigma_s) ** 2)
analytical_llr = np.log(pdf_sig / pdf_bkg)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left Plot: The Gaussian Distributions
axes[0].plot(x_grid, pdf_bkg, label='Background P(x|B)', color='blue', lw=2)
axes[0].plot(x_grid, pdf_sig, label='Signal P(x|S)', color='orange', lw=2)
axes[0].set_title('Toy Gaussian Distributions', fontsize=12)
axes[0].set_xlabel('x', fontsize=11)
axes[0].set_ylabel('Probability Density', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right Plot: Analytical LLR vs Neural Network Learned LLR
axes[1].plot(x_grid, analytical_llr, label='Analytical LLR ln(P_sig/P_bkg)', color='black', linestyle='--', lw=2.5)
axes[1].plot(x_grid, learned_llr, label='Learned LLR (Neural Net)', color='crimson', lw=2)
axes[1].set_title('Likelihood Ratio Estimation via Classifier', fontsize=12)
axes[1].set_xlabel('x', fontsize=11)
axes[1].set_ylabel('Log Likelihood Ratio t(x)', fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()