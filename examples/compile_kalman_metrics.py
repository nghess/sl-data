"""
Compile per-session Kalman decoder metrics into a single master CSV.

Walks {base_path}/{experiment}/kalman_decode/{mouse_id}/{session_id}/metrics.csv
and concatenates all found files, prepending mouse_id and session_id columns.
"""

import csv
import os
from pathlib import Path

BASE_PATH  = "d:/data"
EXPERIMENT = "clickbait-motivate"
OUTPUT_CSV = f"{BASE_PATH}/{EXPERIMENT}/kalman_decode/all_sessions_metrics.csv"


def compile_metrics(base_path: str, experiment: str, output_csv: str) -> None:
    decode_root = Path(base_path) / experiment / "kalman_decode"

    rows = []
    for metrics_file in sorted(decode_root.glob("*/*/metrics.csv")):
        session_id = metrics_file.parent.name
        mouse_id   = metrics_file.parent.parent.name
        with open(metrics_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({'mouse_id': mouse_id, 'session_id': session_id, **row})

    if not rows:
        print(f"No metrics.csv files found under {decode_root}")
        return

    fieldnames = ['mouse_id', 'session_id'] + [
        k for k in rows[0] if k not in ('mouse_id', 'session_id')
    ]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Compiled {len(rows)} sessions → {output_csv}")


if __name__ == "__main__":
    compile_metrics(BASE_PATH, EXPERIMENT, OUTPUT_CSV)
