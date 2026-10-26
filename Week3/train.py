import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import ParameterizedLRClassifier

def train_model():
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # ==========================================
    # 1. GENERATE 4D PARAMETERIZED TRAINING DATA
    # ==========================================
    n_samples = 25000
    mu_b, sigma_b = 0.0, 1.0  # Fixed background

    # Sample random signal parameters (mu_s, sigma_s)
    mu_s_train = np.random.uniform(1.0, 4.0, size=(n_samples, 1))
    sigma_s_train = np.random.uniform(0.8, 2.0, size=(n_samples, 1))
    theta_train = np.hstack([mu_s_train, sigma_s_train])  # Shape: (n_samples, 2)

    # 2D observable x = [x1, x2] for background and signal
    x_bkg_1 = np.random.normal(loc=mu_b, scale=sigma_b, size=(n_samples, 1))
    x_bkg_2 = np.random.normal(loc=mu_b, scale=sigma_b, size=(n_samples, 1))
    x_bkg = np.hstack([x_bkg_1, x_bkg_2])

    x_sig_1 = np.random.normal(loc=mu_s_train, scale=sigma_s_train)
    x_sig_2 = np.random.normal(loc=mu_s_train, scale=sigma_s_train)
    x_sig = np.hstack([x_sig_1, x_sig_2])

    # Combine into 4D input: [x1, x2, mu_s, sigma_s]
    X_bkg_full = np.hstack([x_bkg, theta_train])
    X_sig_full = np.hstack([x_sig, theta_train])

    Y_bkg = np.zeros((n_samples, 1))
    Y_sig = np.ones((n_samples, 1))

    X_train = np.vstack([X_bkg_full, X_sig_full])
    Y_train = np.vstack([Y_bkg, Y_sig])

    # Shuffle dataset
    idx = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[idx], Y_train[idx]

    # Convert to PyTorch tensors & DataLoader
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # ==========================================
    # 2. INITIALIZE MODEL & OPTIMIZER
    # ==========================================
    model = ParameterizedLRClassifier(input_dim=4)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Directory for saving weights
    os.makedirs("checkpoints", exist_ok=True)

    # ==========================================
    # 3. TRAINING LOOP WITH CHECKPOINTING
    # ==========================================
    epochs = 15
    save_every = 3  # Save weights to disk every 3 epochs
    model.train()
    print("Training 4D parameterized neural network...")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

        # Save model weights to disk
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join("checkpoints", f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    print("Training complete and weights saved successfully!\n")

if __name__ == "__main__":
    train_model()