"""
Warmup ablation: compare iterative actor training WITH vs WITHOUT
oracle warmup, for both standard (frozen classifier) and joint
(classifier fine-tuning) modes.

Conditions
----------
  warmup=50   (default) — 50 oracle-only batches before rollout mixing
  warmup=0    (no warmup) — skip oracle pre-training, start with rollouts

Each condition is run across several cost weights.
Results are appended to ABL_warmup.csv via the training script's
save_results_to_csv (method name is tagged so rows are distinguishable).

Usage
-----
  # Run all conditions (standard + joint, with + without warmup):
  python run_warmup_ablation.py

  # Only standard actor (no joint):
  python run_warmup_ablation.py --no-joint

  # Only joint actor (no standard):
  python run_warmup_ablation.py --no-standard

  # Override dataset (default uses ACTOR_DATASET env or config default):
  ACTOR_DATASET=cheears_demog python run_warmup_ablation.py

  # Custom cost weights:
  python run_warmup_ablation.py --cws 0.01 0.005 0.001

  # Custom warmup values to compare:
  python run_warmup_ablation.py --warmups 0 25 50 100

  # Sweep over multiple datasets:
  python run_warmup_ablation.py --datasets womac klg cheears_demog

  # Full sweep: multiple datasets, cost weights, warmups:
  python run_warmup_ablation.py --datasets womac klg --cws 0.01 0.005 --warmups 0 50
"""
import argparse
import os
import subprocess
import sys

# ── defaults ────────────────────────────────────────────────────────────
DEFAULT_CWS = [0.01, 0.005, 0.001]
DEFAULT_WARMUPS = [0, 50]          # with vs without warmup
TOTAL_BATCHES = 1000               # keep total fixed so only warmup varies
CSV_PATH = 'ABL_warmup.csv'


def run(cmd, env):
    """Run a shell command, return exit code."""
    print(f"\n{'=' * 70}")
    print(f"CMD: {cmd}")
    print(f"{'=' * 70}")
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Warmup ablation: with vs without oracle warmup',
    )
    parser.add_argument('--cws', type=float, nargs='+', default=DEFAULT_CWS,
                        help=f'Cost weights to sweep (default: {DEFAULT_CWS})')
    parser.add_argument('--acw', type=float, default=None,
                        help='Auxiliary cost weight (default: same as cw)')
    parser.add_argument('--warmups', type=int, nargs='+', default=DEFAULT_WARMUPS,
                        help=f'Warmup batch counts to compare (default: {DEFAULT_WARMUPS})')
    parser.add_argument('--total_batches', type=int, default=TOTAL_BATCHES,
                        help=f'Total training batches (default: {TOTAL_BATCHES})')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets to sweep over (default: use ACTOR_DATASET env or config default). '
                             'Choices: synthetic, cheears_demog, klg, womac, ILIADD, adni')
    parser.add_argument('--no-joint', action='store_true',
                        help='Skip joint actor experiments')
    parser.add_argument('--no-standard', action='store_true',
                        help='Skip standard (frozen) actor experiments')
    parser.add_argument('--csv', type=str, default=CSV_PATH,
                        help=f'Output CSV path (default: {CSV_PATH})')
    args = parser.parse_args()

    env = os.environ.copy()
    py = sys.executable

    # If --datasets not given, run once with whatever ACTOR_DATASET is set to
    datasets = args.datasets if args.datasets else [None]

    n_runs = 0
    n_fail = 0

    for ds in datasets:
        run_env = env.copy()
        if ds is not None:
            run_env['ACTOR_DATASET'] = ds

        ds_label = ds or run_env.get('ACTOR_DATASET', '(config default)')
        print("\n" + "#" * 70)
        print(f"  DATASET: {ds_label}")
        print("#" * 70)

        # ── Standard actor (frozen classifier) ──────────────────────────
        if not args.no_standard:
            print("\n" + "=" * 70)
            print(f"  STANDARD ACTOR (frozen classifier) — {ds_label}")
            print("=" * 70)
            for warmup in args.warmups:
                for cw in args.cws:
                    acw = args.acw if args.acw is not None else cw
                    tag = f"warmup={warmup}"
                    method_suffix = '_nowarmup' if warmup == 0 else '_warmup'
                    print(f"\n>>> [{ds_label}] [{tag}] cw={cw}  acw={acw}  "
                          f"total={args.total_batches}")

                    cmd = (
                        f"{py} train_actor_iterative.py"
                        f" --cost_weight {cw}"
                        f" --aux_cost_weight {acw}"
                        f" --warmup_batches {warmup}"
                        f" --total_batches {args.total_batches}"
                        f" --csv_path {args.csv}"
                        f" --save_suffix warmup{warmup}"
                        f" --method_suffix {method_suffix}"
                    )
                    rc = run(cmd, run_env)
                    n_runs += 1
                    if rc != 0:
                        n_fail += 1

        # ── Joint actor (classifier fine-tuning) ────────────────────────
        if not args.no_joint:
            print("\n" + "=" * 70)
            print(f"  JOINT ACTOR (classifier fine-tuning) — {ds_label}")
            print("=" * 70)
            for warmup in args.warmups:
                for cw in args.cws:
                    acw = args.acw if args.acw is not None else cw
                    tag = f"warmup={warmup}"
                    method_suffix = '_nowarmup' if warmup == 0 else '_warmup'
                    print(f"\n>>> [{ds_label}] [{tag}] cw={cw}  acw={acw}  "
                          f"total={args.total_batches}")

                    cmd = (
                        f"{py} train_actor_iterative_joint.py"
                        f" --cost_weight {cw}"
                        f" --aux_cost_weight {acw}"
                        f" --warmup_batches {warmup}"
                        f" --total_batches {args.total_batches}"
                        f" --csv_path {args.csv}"
                        f" --save_suffix warmup{warmup}"
                        f" --method_suffix {method_suffix}"
                    )
                    rc = run(cmd, run_env)
                    n_runs += 1
                    if rc != 0:
                        n_fail += 1

    # ── summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Warmup ablation complete: {n_runs} runs, {n_fail} failures")
    print(f"Results: {args.csv}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
