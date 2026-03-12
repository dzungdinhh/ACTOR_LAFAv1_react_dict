"""
Baseline gate ablation: train actors with --baseline all / none.
Matches cost weights from ABL_cls_training.csv ACTOR rows.
Results appended to ABL_baseline_gate.csv (via training script's save_results_to_csv).
"""
import subprocess
import sys

# ACTOR (non-joint) cost weights from ABL_cls_training.csv
ACTOR_CWS = [0.004, 0.003, 0.002, 0.0015, 0.001, 0.0008]

CSV_PATH = 'ABL_baseline_gate.csv'

def run(cmd):
    print(f"\n{'='*60}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, env={**__import__('os').environ, 'ACTOR_DATASET': 'cheears'})
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
    return result.returncode

def main():
    for baseline in ['all', 'none']:
        for cw in ACTOR_CWS:
            # Train non-joint (appends results to CSV automatically)
            run(f"{sys.executable} train_actor_iterative.py "
                f"--cost_weight {cw} --aux_cost_weight {cw} "
                f"--baseline {baseline} --csv_path {CSV_PATH}")

    print(f"\nDone! Check results with: cat {CSV_PATH}")

if __name__ == '__main__':
    main()
