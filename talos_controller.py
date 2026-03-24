import torch

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
# Source of truth now lives in incremental_train.py.
# ==========================================
def compute_loss(pt, pcov, gt):
    from incremental_train import compute_loss as incremental_compute_loss
    return incremental_compute_loss(pt, pcov, gt)
