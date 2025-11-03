#!/bin/bash
# SLURM submission script for CogVideo training (diffusers LoRA / SFT examples)
# Edit the VARIABLES section to match your cluster and job needs.

#########################
# SLURM OPTIONS (change as needed)
#########################
#SBATCH --job-name=cogvideo_train
#SBATCH --partition=gpu                 # partition/queue name
#SBATCH --nodes=1                       # number of nodes (set >1 for multi-node)
#SBATCH --ntasks-per-node=1             # usually 1 (we use torchrun for per-node GPUs)
#SBATCH --cpus-per-task=12              # CPUs for dataloading & preprocessing
#SBATCH --gres=gpu:4                    # GPUs per node
#SBATCH --mem=200G                      # memory per node
#SBATCH --time=48:00:00                 # HH:MM:SS
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=your.email@domain

#########################
# USER CONFIGURABLE VARIABLES
#########################
# Training config
GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"              # change for multi-node runs
MASTER_PORT=12345
TRAIN_SCRIPT="finetune/train.py"     # or use provided bash launchers in finetune/
CONFIGS="finetune/configs/zero2.yaml" # example config(s) to pass
OUTPUT_DIR="./outputs/train_${SLURM_JOB_ID}"
EXTRA_ARGS="--model_name cogvideox1_5 --training_type lora" # extra args for your train.py/launch

# Environment activation (edit to your environment)
CONDA_ACTIVATE_CMD="source /path/to/conda.sh; conda activate cogvideo"

#########################
# Prepare environment
#########################
set -euo pipefail
mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# Activate environment
if [ -n "${CONDA_ACTIVATE_CMD}" ]; then
  eval "${CONDA_ACTIVATE_CMD}"
fi

# Export some useful vars for distributed training
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK}}

# If multi-node, SLURM provides useful env vars (SLURM_NODELIST, SLURM_NTASKS)
if [ ${NNODES} -gt 1 ]; then
  # For multi-node, you must set MASTER_ADDR and MASTER_PORT appropriately.
  # Example using first node as master (simple approach):
  MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
  export MASTER_ADDR
  export MASTER_PORT=${MASTER_PORT}
  echo "Multi-node run: MASTER_ADDR=${MASTER_ADDR}, NNODES=${NNODES}"
fi

#########################
# Launch training
#########################
# Option A: Use torchrun to spawn processes per GPU on this node (recommended for single-node multi-GPU)
LAUNCH_CMD=(python -u -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} \
  --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  ${TRAIN_SCRIPT} ${CONFIGS} --output_dir ${OUTPUT_DIR} ${EXTRA_ARGS})

# Option B: Use provided shell wrappers (e.g., finetune/train_ddp_t2v.sh) — uncomment and edit the script instead
# LAUNCH_CMD=(bash finetune/train_ddp_t2v.sh)

# Option C: If you use deepspeed, launch via deepspeed: (edit deepspeed config accordingly)
# LAUNCH_CMD=(deepspeed --num_gpus=${GPUS_PER_NODE} finetune/train.py --base ${CONFIGS} ...)

# Print and run
echo "Running training with: ${LAUNCH_CMD[@]}"
"${LAUNCH_CMD[@]}"

echo "Training finished. Outputs: ${OUTPUT_DIR}"
