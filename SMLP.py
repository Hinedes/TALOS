"""
SMLP.py - Phase-Aware SpectralMLP (Rev. 4 - Dual Trunk)
TALOS NIO Neural Backbone

Architecture:
    - Wrapper: Handles CPU-side FFT.
    - Phase-Aware Extraction: Separates Real and Imaginary components to preserve
        the sign of the DC bin (Gravity projection/Turn direction) and temporal phase.
    - Core: Fully decoupled Dual-Trunk MLP to prevent Negative Transfer between
        kinematic gradients and uncertainty NLL gradients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralMLPNPU(nn.Module):
    def __init__(self):
        super().__init__()
        # --- TRANSLATION PATHWAY (Dedicated to pure kinematics) ---
        # 396 inputs -> 256 -> 128 -> 64 -> 3
        self.fc1_t = nn.Linear(396, 256)
        self.bn1_t = nn.BatchNorm1d(256)
        self.fc2_t = nn.Linear(256, 128)
        self.bn2_t = nn.BatchNorm1d(128)
        self.fc3_t = nn.Linear(128, 64)
        self.bn3_t = nn.BatchNorm1d(64)
        self.head_trans = nn.Linear(64, 3)

        # --- COVARIANCE PATHWAY (Dedicated to NLL uncertainty) ---
        # 396 inputs -> 256 -> 128 -> 64 -> 3
        self.fc1_c = nn.Linear(396, 256)
        self.bn1_c = nn.BatchNorm1d(256)
        self.fc2_c = nn.Linear(256, 128)
        self.bn2_c = nn.BatchNorm1d(128)
        self.fc3_c = nn.Linear(128, 64)
        self.bn3_c = nn.BatchNorm1d(64)
        self.head_cov = nn.Linear(64, 3)

        # Shared dropout rate, applied independently to each trunk
        self.drop = nn.Dropout(0.15)

        # Initialize the covariance head to predict a reasonable starting variance
        nn.init.normal_(self.head_cov.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head_cov.bias, -2.0)

    def forward(self, x):
        # Translation Forward Pass
        t = F.relu(self.bn1_t(self.fc1_t(x)))
        t = self.drop(t)
        t = F.relu(self.bn2_t(self.fc2_t(t)))
        t = F.relu(self.bn3_t(self.fc3_t(t)))
        pred_vel = self.head_trans(t)

        # Covariance Forward Pass
        c = F.relu(self.bn1_c(self.fc1_c(x)))
        c = self.drop(c)
        c = F.relu(self.bn2_c(self.fc2_c(c)))
        c = F.relu(self.bn3_c(self.fc3_c(c)))
        pred_cov = self.head_cov(c)

        return pred_vel, pred_cov


class SpectralMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.npu_core = SpectralMLPNPU()

    def forward(self, x_raw):
        B = x_raw.size(0)
        fft_c = torch.fft.rfft(x_raw, dim=-1)

        real_part = fft_c.real
        imag_part = fft_c.imag

        real_scaled = torch.sign(real_part) * torch.log1p(torch.abs(real_part))
        imag_scaled = torch.sign(imag_part) * torch.log1p(torch.abs(imag_part))

        x_spec = torch.cat([real_scaled, imag_scaled], dim=-1).view(B, -1)
        return self.npu_core(x_spec)


# Alias for backwards compatibility with incremental_train.py imports
BigSpectralMLP = SpectralMLP


if __name__ == '__main__':
    model = SpectralMLP()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randn(4, 6, 64)
    t, lv = model(dummy)
    print(f"Translation : {t.shape}")
    print(f"LogVar      : {lv.shape}")