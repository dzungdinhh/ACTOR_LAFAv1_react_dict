"""
Run the full ACTOR-LAFA pipeline:
  1. Train classifier
  2. Generate oracle rollouts
  3. Train actor (iterative)
  4. Evaluate actor
"""
import argparse
import subprocess
import sys
import os


STEPS = ['classifier', 'oracle', 'actor', 'evaluate']


def run_step(script, env, extra_args=None):
    """Run a Python script as a subprocess."""
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='Run the full ACTOR-LAFA pipeline',
    )
    parser.add_argument('--data', type=str, default=None,
                        help="Dataset: 'synthetic', 'cheears_demog', 'klg', or 'womac' "
                             "(overrides config.py default)")
    parser.add_argument('--cw', type=float, default=None,
                        help='cost_weight for oracle / actor / evaluation')
    parser.add_argument('--acw', type=float, default=None,
                        help='aux_cost_weight for oracle / actor / evaluation')
    parser.add_argument('--joint', action='store_true', default=False,
                        help='Use joint iterative actor (with classifier fine-tuning and mask augmentation)')
    parser.add_argument('--skip', nargs='*', default=[], choices=STEPS,
                        help='Steps to skip (e.g. --skip classifier oracle)')
    args = parser.parse_args()

    # Build env with optional dataset override
    env = os.environ.copy()
    if args.data is not None:
        env['ACTOR_DATASET'] = args.data
        print(f"Dataset override: {args.data}")

    # Shared cost args for scripts that accept them
    cost_args = []
    if args.cw is not None:
        cost_args += ['--cost_weight', str(args.cw)]
    if args.acw is not None and args.cw is not None and args.acw != args.cw:
        raise ValueError(f"Expected cw == acw, got cw={args.cw} and acw={args.acw}")
    if args.cw is not None:
        cost_args += ['--aux_cost_weight', str(args.cw)]

    # 1. Train classifier
    if 'classifier' not in args.skip:
        run_step('train_classifier.py', env)

    # 2. Generate oracle rollouts
    if 'oracle' not in args.skip:
        run_step('generate_oracle.py', env, cost_args)

    # 3. Train actor
    if 'actor' not in args.skip and not args.joint:
        run_step('train_actor_iterative.py', env, cost_args)
    elif 'actor' not in args.skip and args.joint:
        run_step('train_actor_iterative_joint.py', env, cost_args)

    # 4. Evaluate
    if 'evaluate' not in args.skip:
        eval_args = cost_args + (['--joint'] if args.joint else [])
        run_step('evaluate.py', env, eval_args)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
