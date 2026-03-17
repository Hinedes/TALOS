import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def append_eval_csv(run_dir: Path, round_num: int,
                    summary_row: dict,
                    step_rows: list[dict],
                    update_rows: list[dict]) -> Path:
    """Append one round of evaluation telemetry into a single run-level CSV.

    Output schema is long-format with `row_type` in {summary, step, update}.
    This allows all logging and plot-source data to be captured in one file
    for downstream agents and analytics.
    """
    out_csv = run_dir / 'talos_eval_log.csv'

    blocks = []

    s = summary_row.copy()
    s['round'] = round_num
    s['row_type'] = 'summary'
    blocks.append(pd.DataFrame([s]))

    if step_rows:
        step_df = pd.DataFrame(step_rows)
        step_df['round'] = round_num
        step_df['row_type'] = 'step'
        blocks.append(step_df)

    if update_rows:
        upd_df = pd.DataFrame(update_rows)
        upd_df['round'] = round_num
        upd_df['row_type'] = 'update'
        blocks.append(upd_df)

    out_df = pd.concat(blocks, ignore_index=True, sort=False)
    out_df.to_csv(out_csv, mode='a', header=not out_csv.exists(), index=False)
    return out_csv

def generate_diagnostic_dashboard(diag_v_pred_local, diag_v_gt_local, diag_mahal_sq, 
                                 diag_v_gt_mag, diag_pred_std, diag_abs_error,
                                 round_num, run_dir, slap_threshold=5.0):
    # Convert lists to numpy arrays for easier indexing
    v_pred = np.array(diag_v_pred_local)
    v_gt = np.array(diag_v_gt_local)
    mahal = np.array(diag_mahal_sq)
    gt_mag = np.array(diag_v_gt_mag)
    std = np.array(diag_pred_std)
    error = np.array(diag_abs_error)

    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f'TALOS Diagnostic Lens - Round {round_num}', fontsize=20, fontweight='bold')

    # --- LENS 1: Scale Collapse (Forward/Y-Axis Focus) ---
    # We focus on the Y-axis (index 1) as it's the primary axis of human travel
    axs[0].scatter(v_gt[:, 1], v_pred[:, 1], alpha=0.5, s=10, c='cobalt' if 'cobalt' in plt.colormaps else 'blue')
    lims = [np.min([v_gt[:, 1], v_pred[:, 1]]), np.max([v_gt[:, 1], v_pred[:, 1]])]
    axs[0].plot(lims, lims, 'r--', alpha=0.75, zorder=3, label='Ideal (y=x)')
    axs[0].set_title('Lens 1: Scale Collapse (Local Forward Velocity)', fontsize=14)
    axs[0].set_xlabel('Ground Truth (m/s)')
    axs[0].set_ylabel('Neural Prediction (m/s)')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # --- LENS 2: Filter Tension (Slap Timeline) ---
    time_axis = np.arange(len(mahal))
    slap_threshold_sq = slap_threshold ** 2
    axs[1].plot(time_axis, mahal, color='black', alpha=0.6, label='Mahal Distance²')
    axs[1].fill_between(time_axis, 0, mahal, where=(mahal > slap_threshold_sq), color='red', alpha=0.3,
                        label=f'SLAP (Threshold {slap_threshold:.1f}²)')
    ax2_twin = axs[1].twinx()
    ax2_twin.plot(time_axis, gt_mag, color='green', alpha=0.4, label='GT Speed (m/s)')
    axs[1].set_title('Lens 2: Slap Gate Timeline / Tension', fontsize=14)
    axs[1].set_xlabel('Neural Update Index')
    axs[1].set_ylabel('Mahalanobis Sq')
    ax2_twin.set_ylabel('Speed (m/s)')
    axs[1].grid(True, alpha=0.3)
    # Combine legends for twin axes
    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axs[1].legend(lines + lines2, labels + labels2, loc='upper right')

    # --- LENS 3: Covariance Shadowing ---
    # Compare mean error vs mean predicted uncertainty across all 3 axes
    mean_err = np.mean(error, axis=1)
    mean_std = np.mean(std, axis=1)
    axs[2].plot(time_axis, mean_err, label='Actual Abs Error', alpha=0.7)
    axs[2].plot(time_axis, mean_std, label='Predicted StdDev (Shadow)', alpha=0.7, linestyle='--')
    axs[2].set_title('Lens 3: Covariance Shadowing Health', fontsize=14)
    axs[2].set_xlabel('Neural Update Index')
    axs[2].set_ylabel('Magnitude (m/s)')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(run_dir / f'diagnostic_round_{round_num}.png', dpi=150)
    plt.close()