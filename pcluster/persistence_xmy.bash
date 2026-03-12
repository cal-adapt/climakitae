#!/bin/bash -l

#SBATCH -J test
#SBATCH --array=1-15
#SBATCH -N 1
#SBATCH -o stdout/persistent_XMY/%A-%a.out
#SBATCH -e err/persistent_XMY/%A-%a.err
#SBATCH -c 12
#SBATCH --mem=72G

# Define variables
NOTEBOOK_IN=notebooks/persistence_XMY.ipynb
NOTEBOOK_OUT=notebooks_outputs/output-${SLURM_ARRAY_TASK_ID}.ipynb

# Read in list of parameters (one line per script run):
ARG_ID=$(cat input-list.dat | awk "NR==$SLURM_ARRAY_TASK_ID")

log_file="stdout/persistent_XMY/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out"

# Print SBATCH job settings for debugging to the log file
{
  echo "====================================="
  echo "Network: $NETWORK"
  echo "Job Name: $SLURM_JOB_NAME"
  echo "Job ID: $SLURM_JOB_ID"
  echo "Partition: $SLURM_JOB_PARTITION"
  echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
  echo "Tasks Per Node: $SLURM_NTASKS_PER_NODE"
  echo "Total Tasks: $SLURM_NTASKS"
  echo "CPUs Per Task: $SLURM_CPUS_PER_TASK"
  echo "Job Start Time: $(date)"
  echo "====================================="
} >> "$log_file"

echo "Running job: Task ID=$SLURM_ARRAY_TASK_ID, Args=${ARG_ID}"

# Run with papermill using conda
conda run -n ae papermill \
  "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" \
  -p args "${ARG_ID}"

# Calculate and print elapsed time to the log file
elapsed_time=$((end_time - start_time))
{
  echo "====================================="
  echo "Job completed in $elapsed_time seconds."
  echo "Job End Time: $(date)"
  echo "====================================="
} >> "$log_file"