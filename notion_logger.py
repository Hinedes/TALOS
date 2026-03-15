"""
notion_logger.py -- Appends training run results to TALOS Research Log.
Reads NOTION_TOKEN from .env file in the same directory, or from environment.
Usage: python3 notion_logger.py --ate 2.908 --round 9 --total 22
"""
import os
import argparse
import requests
from datetime import date
from pathlib import Path

RESEARCH_LOG_ID = "bd71957c581a4c3985a9332a5d7c6619"

def _load_token():
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith('NOTION_TOKEN='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('NOTION_TOKEN')

def log_run(best_ate: float, best_round: int, total_rounds: int):
    token = _load_token()
    if not token:
        print("[notion_logger] NOTION_TOKEN not found -- skipping.")
        return False

    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    body = {
        "children": [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content":
                    f"{date.today()} — Training Run: {best_ate:.3f}m ATE @ Round {best_round}"
                }}]}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content":
                    f"Best ATE: {best_ate:.3f}m | Achieved: Round {best_round}/{total_rounds} | "
                    f"Val: shelby_arroyo (300s walking) | Checkpoint: golden/talos_best_physical.pth"
                }}]}
            },
            {
                "object": "block",
                "type": "divider",
                "divider": {}
            }
        ]
    }
    r = requests.patch(
        f"https://api.notion.com/v1/blocks/{RESEARCH_LOG_ID}/children",
        headers=headers, json=body, timeout=10
    )
    if r.status_code == 200:
        print("[notion_logger] Research Log updated.")
        return True
    else:
        print(f"[notion_logger] Failed: {r.status_code} {r.text}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ate",   type=float, required=True)
    parser.add_argument("--round", type=int,   required=True)
    parser.add_argument("--total", type=int,   required=True)
    args = parser.parse_args()
    log_run(args.ate, args.round, args.total)
