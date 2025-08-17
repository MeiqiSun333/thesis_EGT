import os
import argparse
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np
from SALib.sample import morris as morris_sample
from Simulate import simulate


def generate_or_load_params(out_dir, seed, problem):
    out_dir.mkdir(parents=True, exist_ok=True)
    pv_path = out_dir / 'param_values.csv'
    if pv_path.exists():
        param_values = np.loadtxt(pv_path, delimiter=",")
    else:
        np.random.seed(seed)
        param_values = morris_sample.sample(problem, N=10, num_levels=4)
        np.savetxt(pv_path, param_values, delimiter=',')
    return param_values


def expected_files(raw_dir, idx, rep):
    agent_path = raw_dir / f'agent_data_{idx}_{rep}.csv'
    model_path = raw_dir / f'model_data_{idx}_{rep}.csv'
    return agent_path, model_path


def run_one(param_idx, params, num_steps, rep, raw_dir, seed_sim=None):
    if seed_sim is not None:
        seed = (int(seed_sim) + param_idx * 1_000_003 + rep * 997) & 0xFFFFFFFF
        np.random.seed(seed)

    rewiring_p, alpha, rat, risk, normalize, dependence, dependence_game = params
    sim_kwargs = {
        "N": 200,
        "rewiring_p": float(rewiring_p),
        "alpha": float(alpha),
        "rat": float(rat),
        "rounds": 1,
        "steps": int(num_steps),
        "alwaysOwn": False,
        "alwaysSafe": False,
        "UV": (True, None, None, False),
        "risk_distribution": "default",
        "utility_function": "prospect",
        "normalizeGames": float(normalize),
        "risk": float(risk),
        "dependence": float(dependence),
        "dependence_game": float(dependence_game),
    }

    model_df, agent_df = simulate(**sim_kwargs)

    agent_path, model_path = expected_files(raw_dir, param_idx, rep)
    agent_tmp = agent_path.with_suffix('.csv.tmp')
    model_tmp = model_path.with_suffix('.csv.tmp')

    agent_df.to_csv(agent_tmp, index=False)
    model_df.to_csv(model_tmp, index=False)
    os.replace(agent_tmp, agent_path)
    os.replace(model_tmp, model_path)
    
    return (True, param_idx, rep)


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    parser = argparse.ArgumentParser(description="Run local simulations and dump raw CSVs.")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps per simulation")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions per parameter set")
    parser.add_argument("--out_dir", type=str, default="Data/GSA", help="Output directory")
    parser.add_argument("--seed_params", type=int, default=42,
                        help="Seed for generating param_values.csv")
    parser.add_argument("--seed_sim", type=int, default=None,
                        help="Base seed for simulation RNG (optional)")
    # set the workers here
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers. 0 for auto (CPU count - 1), 1 for sequential.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing simulation files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    raw_dir = out_dir / 'raw_runs'
    raw_dir.mkdir(parents=True, exist_ok=True)

    problem = {
        'num_vars': 7,
        'names': ['rewiring_p', 'alpha', 'rat', 'risk', 'normalize', 'dependence', 'dependence_game'],
        'bounds': [[0.1, 0.9],
                   [0.0, 1.0],
                   [0.0, 2.0],
                   [0.0, 2.0],
                   [0.0, 1.0],
                   [0.0, 1.0],
                   [0.0, 1.0]]
    }

    param_values = generate_or_load_params(out_dir, args.seed_params, problem)

    tasks = []
    skipped = 0
    for idx, params in enumerate(param_values):
        for rep in range(int(args.repetitions)):
            a_path, m_path = expected_files(raw_dir, idx, rep)
            if (not args.overwrite) and a_path.exists() and m_path.exists():
                skipped += 1
                continue
            tasks.append((idx, params, args.steps, rep, raw_dir, args.seed_sim))

    total = len(param_values) * int(args.repetitions)
    print(f'Total tasks={total}, To run={len(tasks)}, Skipped={skipped}, Output dir={out_dir.resolve()}')

    if not tasks:
        print('Nothing to run!')
        return
        
    start_time = time.time()

    if args.workers == 1:
        n_workers = 1
    elif args.workers <= 0:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_workers = args.workers

    if n_workers == 1:
        print("Running in sequential mode (1 worker)...")
        done, fail = 0, 0
        for t in tasks:
            try:
                run_one(*t)
                done += 1
                print(f'\r[SUCCESS] param_set={t[0]} rep={t[3]} | Progress: {done}/{len(tasks)}', end="", flush=True)
            except Exception as e:
                fail += 1
                print(f'\n[ERROR] param_set={t[0]} rep={t[3]}: {e}')
        print()
    else:
        print(f"Starting parallel execution with {n_workers} workers...")
        done, fail = 0, 0
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {executor.submit(run_one, *t): t for t in tasks}

            for future in as_completed(future_to_task):
                original_task = future_to_task[future]
                param_idx, _, _, rep, _, _ = original_task
                try:
                    result, p_idx, r_idx = future.result()
                    if result:
                        done += 1
                        print(f'\r Progress: {done+fail}/{len(tasks)} | Last success: param_set={p_idx} rep={r_idx}', end="", flush=True)
                except Exception as exc:
                    fail += 1
                    print(f'\n [ERROR] on param_set={param_idx} rep={rep}: {exc}')
        print()

    end_time = time.time()
    duration = end_time - start_time
    print(f'\n All done! Success: {done}, Failed: {fail}. Total time: {duration:.2f} seconds.')


if __name__ == "__main__":
    # Run by entering the command 'python run_psa.py --repetitions 10 --steps 100' in the terminal
    main()