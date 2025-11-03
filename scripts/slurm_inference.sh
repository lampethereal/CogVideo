#!/bin/bash
# SLURM submission script for CogVideo inference (diffusers pipeline)
# Edit the VARIABLES section below to match your cluster and job needs.

#########################
# SLURM OPTIONS (change as needed)
#########################
#SBATCH --job-name=cogvideo_infer
#SBATCH --partition=gpu                 # partition/queue name
#SBATCH --gres=gpu:1                    # number of GPUs (per node)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=8               # CPUs per task (useful for data loading)
#SBATCH --mem=64G                       # memory per node (or use --mem-per-cpu)
#SBATCH --time=04:00:00                 # HH:MM:SS
#SBATCH --output=logs/inference-%j.out  # STDOUT
#SBATCH --error=logs/inference-%j.err   # STDERR
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=your.email@domain  # set your email if you want notifications

#########################
# USER CONFIGURABLE VARIABLES
#########################
# model / data / output paths (change these)
MODEL_PATH="THUDM/CogVideoX1.5-5b"    # HF model id or local path
LORA_PATH=""                          # optional: path to lora weights directory
PROMPT="A serene sunrise over a mountain lake"  # example prompt
OUTPUT_PATH="./outputs/inference_${SLURM_JOB_ID}.mp4"
NUM_FRAMES=81
NUM_STEPS=50
GUIDANCE_SCALE=6.0
GENERATE_TYPE="t2v"                   # t2v / i2v / v2v
DTYPE="bfloat16"                     # bfloat16 or float16
IMAGE_OR_VIDEO_PATH=""               # for i2v/v2v
FPS=16

# (Optional) path to python environment activation script
CONDA_ACTIVATE_CMD="source /path/to/conda.sh; conda activate cogvideo" # or use your environment activation command

#########################
# Prepare environment
#########################
set -euo pipefail
mkdir -p logs

# Activate environment (edit to your cluster's env setup)
if [ -n "${CONDA_ACTIVATE_CMD}" ]; then
  eval "${CONDA_ACTIVATE_CMD}"
fi

# Show some info
echo "Job id: ${SLURM_JOB_ID}"
echo "Running on nodes: $(scontrol show hostnames $SLURM_NODELIST)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#########################
# Inference command
#########################
# We use the diffusers CLI demo. Adjust flags as necessary.
INFER_CMD=(python -u inference/cli_demo.py
  --prompt "${PROMPT}"
  --model_path "${MODEL_PATH}"
  --num_frames ${NUM_FRAMES}
  --num_inference_steps ${NUM_STEPS}
  --output_path "${OUTPUT_PATH}"
  --guidance_scale ${GUIDANCE_SCALE}
  --generate_type "${GENERATE_TYPE}"
  --dtype "${DTYPE}"
  --fps ${FPS}
)

# Add optional image/video or LoRA args
if [ -n "${IMAGE_OR_VIDEO_PATH}" ]; then
  INFER_CMD+=(--image_or_video_path "${IMAGE_OR_VIDEO_PATH}")
fi
if [ -n "${LORA_PATH}" ]; then
  INFER_CMD+=(--lora_path "${LORA_PATH}")
fi

# Print and run
echo "Running inference command: ${INFER_CMD[@]}"
"${INFER_CMD[@]}"

echo "Inference finished. Output: ${OUTPUT_PATH}"
