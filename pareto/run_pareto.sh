#!/bin/bash
#SBATCH --job-name=pareto_explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00             # 根据单个模拟的耗时调整，可以设短一些
#SBATCH --mem=4G
# 数组大小必须是 N * n_repetitions - 1。这里是 1024 * 1 - 1 = 1023
#SBATCH --array=0-1023
#SBATCH --output=slurm_logs/slurm_job_%A_task_%a.out
#SBATCH --error=slurm_logs/slurm_job_%A_task_%a.err

set -euo pipefail

cleanup_and_rescue() {
    echo "--- Rescue function triggered ---"
    if [ -d "$TMPDIR_RESULTS" ]; then
        echo "Rescuing data from $TMPDIR_RESULTS to $FINAL_RESULT_DIR"
        rsync -av "$TMPDIR_RESULTS/" "$FINAL_RESULT_DIR/"
    else
        echo "No results directory found in TMPDIR. Nothing to rescue."
    fi
    echo "--- Rescue complete ---"
}
trap cleanup_and_rescue EXIT

echo "Job begins at $(hostname) on $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
export PYTHONUSERBASE=$HOME/.local
export PATH=$PYTHONUSERBASE/bin:$PATH


FINAL_RESULT_DIR="$HOME/Data/pareto"
SOURCE_DIR="$HOME"
TMPDIR_RESULTS="$TMPDIR/results"

mkdir -p "$FINAL_RESULT_DIR"
mkdir -p slurm_logs

echo "Copying source files to $TMPDIR"
cp "$SOURCE_DIR"/run_pareto.py "$TMPDIR/"
cp "$SOURCE_DIR"/Simulate.py "$TMPDIR/"
cp "$SOURCE_DIR"/GamesModel.py "$TMPDIR/" 
cp "$SOURCE_DIR"/GameAgent.py "$TMPDIR/"
cp "$SOURCE_DIR"/Game.py "$TMPDIR/"
cp "$SOURCE_DIR"/config.py "$TMPDIR/"
cp "$SOURCE_DIR"/utils.py "$TMPDIR/"

cd "$TMPDIR"
mkdir -p "$TMPDIR_RESULTS"

echo "Running Python script for task ID: $SLURM_ARRAY_TASK_ID"
python -u run_pareto.py "$SLURM_ARRAY_TASK_ID" "$TMPDIR_RESULTS"

echo "Python script finished for task ID: $SLURM_ARRAY_TASK_ID"
echo "Job finished at $(date)"