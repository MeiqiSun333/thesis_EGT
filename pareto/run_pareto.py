import os
import sys
import numpy as np
from SALib.sample import latin
from Simulate import simulate

def main():
    """
    HPC task script for exploring a 7D parameter space using Latin Hypercube Sampling.
    """

    problem = {
        'num_vars': 7,
        'names': ['rewiring_p', 'alpha', 'rat', 'risk', 'normalizeGames', 'dependence', 'dependence_game'],
        'bounds': [[0.1, 0.9],
                   [0.0, 1.0],
                   [0.0, 2.0],
                   [0.0, 2.0],
                   [0.1, 0.9],
                   [0.1, 0.9],
                   [0.1, 0.9]]
    }

    # 设置要生成的参数组合总数
    # 这个 N 必须与你的 SLURM 脚本中的 --array=0-(N-1) 保持一致
    N = 1024  # 例如，我们生成1024个不同的参数组合

    seed = 42
    

    param_values = latin.sample(problem, N=N, seed=seed)
    

    n_repetitions = 1


    try:
        task_id = int(sys.argv[1])
        output_dir = sys.argv[2]
        

        rep = task_id // N
        param_index = task_id % N

        if rep >= n_repetitions:
            print(f"Error: Repetition index {rep} is out of bounds for Task ID {task_id}.")
            sys.exit(1)

    except (IndexError, ValueError):
        print("Error: Invalid command-line arguments.")
        print(f"Usage: python {sys.argv[0]} <task_id> <output_directory>")
        sys.exit(1)

    current_params_array = param_values[param_index]
    
    current_params_dict = dict(zip(problem['names'], current_params_array))

    print(f"--- Running Task ID: {task_id} ---")
    print(f"Repetition: {rep}")
    print("Parameters:")
    for name, value in current_params_dict.items():
        print(f"  {name}: {value:.4f}")

    param_str_parts = [f"{name}_{value:.3f}" for name, value in current_params_dict.items()]

    param_str = "_".join(param_str_parts)
    
    model_filename = f"model_data_{param_str}_rep{rep}.csv"
    model_filepath = os.path.join(output_dir, model_filename)

    if os.path.exists(model_filepath):
        print(f"Result file already exists. Skipping task {task_id}.")
        sys.exit(0)


    sim_kwargs = current_params_dict.copy()
    sim_kwargs.update({
        "N": 200,
        "rounds": 1,
        "steps": 200,
        "alwaysOwn": False,
        "alwaysSafe": False,
        "UV": (True, None, None, False),
        "risk_distribution": "default",
        "utility_function": "prospect"
    })
    

    for key in sim_kwargs:
        if isinstance(sim_kwargs[key], np.generic):
            sim_kwargs[key] = sim_kwargs[key].item()

    print("Simulating...")
    model_data, _ = simulate(**sim_kwargs)

    model_data.to_csv(model_filepath, index=False)

    print(f"Successfully finished task {task_id}. Data saved to {model_filepath}")

if __name__ == "__main__":
    main()