import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def monitor_magnitude_cure(csv_path):
    # Load with low_memory=False to gracefully handle the mixed types in step/update rows
    df = pd.read_csv(csv_path, low_memory=False)
    
    # CRITICAL: Isolate the round-level summary data from the high-frequency step data
    if 'row_type' in df.columns:
        df = df[df['row_type'] == 'summary'].copy()
        
    # Ensure the dataframe is sorted by round just in case
    if 'round' in df.columns:
        df = df.sort_values('round')
        rounds = df['round']
    else:
        rounds = df.index + 1
        
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('TALOS Magnitude Anchoring Diagnostics', fontweight='bold', fontsize=14)
        
    # 1. Speed Ratio
    if 'pred_gt_speed_ratio' in df.columns:
        axs[0].plot(rounds, df['pred_gt_speed_ratio'], marker='o', color='tab:blue', linewidth=2)
        axs[0].axhline(1.0, color='black', linestyle='--', alpha=0.7, label='Ideal Target')
        axs[0].set_ylabel('Speed Ratio (Pred/GT)')
        axs[0].set_title('Velocity Magnitude Calibration')
        axs[0].legend()
    
    # 2. Gyro Bias Z
    if 'gyro_bias_z' in df.columns:
        axs[1].plot(rounds, df['gyro_bias_z'], marker='s', color='tab:red', linewidth=2)
        axs[1].axhline(0.0, color='black', linestyle='--', alpha=0.7, label='Zero Bias')
        axs[1].set_ylabel('Gyro Bias Z (rad/s)')
        axs[1].set_title('Yaw Bias Observability')
        axs[1].legend()
    
    # 3. Yaw Error
    if 'yaw_err_mean_deg' in df.columns:
        axs[2].plot(rounds, df['yaw_err_mean_deg'], marker='^', color='tab:orange', linewidth=2)
        axs[2].set_ylabel('Mean Yaw Err (deg)')
        axs[2].set_title('Heading Error Drift')
    
    # 4. ATE
    if 'mean_ate_m' in df.columns:
        axs[3].plot(rounds, df['mean_ate_m'], marker='d', color='tab:green', linewidth=2)
        axs[3].set_ylabel('Mean ATE (m)')
        axs[3].set_title('Overall Physical Drift')
    
    axs[3].set_xlabel('Training Round', fontweight='bold')
    axs[3].set_xlim(left=0)
    
    plt.tight_layout()
    
    output_file = Path(csv_path).parent / 'magnitude_diagnostics.png'
    plt.savefig(output_file, dpi=150)
    print(f":: Diagnostic plot generated -> {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot TALOS magnitude anchoring diagnostics.")
    parser.add_argument('csv_path', type=str, help="Path to the telemetry summary CSV")
    args = parser.parse_args()
    
    try:
        monitor_magnitude_cure(args.csv_path)
    except FileNotFoundError:
        print(f"!! Error: Could not find file at {args.csv_path}")
    except Exception as e:
        print(f"!! Error processing CSV: {e}")