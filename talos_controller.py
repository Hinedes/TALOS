import torch
import torch.nn.functional as F

# ==========================================
# ESKF FUSION THRESHOLDS (Mutation Target)
# ==========================================
SLAP_THRESHOLD       = 3.5
R_OBS_MIN_DIAG       = 0.05
R_OBS_MAX_DIAG       = 2.00
USE_DYNAMIC_R_OBS    = True
R_OBS_FIXED_DIAG     = 0.10
PRED_VEL_GAIN        = 1.00

# ==========================================
# NEURAL LOSS TOPOLOGY (Mutation Target)
# ==========================================
def compute_loss(pt, pcov, gt):
    var = torch.exp(pcov)

    # 1. The Translation Head: Robust, High-Weight Loss
    pred_norm = pt.norm(dim=-1)
    gt_norm = gt.norm(dim=-1)

    # Direction Loss (Dominant, Scale-Invariant)
    loss_dir = (1.0 - F.cosine_similarity(pt, gt, dim=-1, eps=1e-8)).unsqueeze(-1)

    # Magnitude Loss (Strong Anchor, Outlier-Robust)
    mask = gt_norm > 0.05
    if mask.any():
        speed_ratio = pred_norm[mask] / gt_norm[mask]
        loss_mag_raw = F.huber_loss(speed_ratio, torch.ones_like(speed_ratio), delta=0.15, reduction='none')
        loss_mag = torch.zeros_like(gt_norm)
        loss_mag[mask] = loss_mag_raw
    else:
        loss_mag = torch.zeros_like(gt_norm)

    # 2. The Covariance Head: NLL on Detached Predictions
    mse_detached = (pt.detach() - gt) ** 2
    loss_nll = 0.5 * (pcov + mse_detached / var)

    # 3. Independent Weighting Strategy
    weight = 1.0 + 10.0 * gt_norm.unsqueeze(-1)
    loss_covariance = torch.mean(weight * loss_nll)

    lambda_dir = 2.0
    lambda_mag = 1.0
    loss_velocity = lambda_dir * torch.mean(loss_dir) + lambda_mag * torch.mean(loss_mag)

    return loss_velocity + loss_covariance
