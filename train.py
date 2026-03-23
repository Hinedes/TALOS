import subprocess
import pandas as pd
from pathlib import Path


def get_latest_run_dir(golden_dir: Path) -> Path | None:
    run_dirs = [p for p in golden_dir.glob("run_*") if p.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def run_eval():
    # 1. Execute the incremental training on CPU to save VRAM for OmniClaw
    # We use --seed 1337 for reproducible experiments
    cmd = ["python", "incremental_train.py", "--seed", "1337"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"TRAINING_CRASHED\n{result.stderr}")
        return

    # 2. Find the latest run directory dynamically to grab the CSV
    latest_run = get_latest_run_dir(Path("golden"))
    if latest_run is None:
        print("ERROR: No run directory found.")
        return

    latest_csv = latest_run / "talos_eval_log.csv"
    
    # 3. Parse the last summary row for the agent's reward signal
    df = pd.read_csv(latest_csv)
    summary = df[df['row_type'] == 'summary'].iloc[-1]
    
    # 4. Print the metrics the agent needs to see
    print(f"--- EVALUATION_COMPLETE ---")
    print(f"ATE: {summary['caged_ate_m']:.4f}")
    print(f"Speed_Ratio: {summary['pred_gt_speed_ratio']:.4f}")
    print(f"Cos_Sim: {summary['cos_sim_mean']:.4f}")
    print(f"Slap_Rate: {summary['slap_rate_pct']:.2f}%")
    print(f"Yaw_Drift: {summary['yaw_err_mean_deg']:.2f}")

if __name__ == "__main__":
    run_eval()