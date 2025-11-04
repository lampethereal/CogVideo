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


#########################
# USER CONFIGURABLE VARIABLES
#########################
# 📝 详细参数说明请查看: docs/INFERENCE_GUIDE_CN.md

# === 模型配置 ===
MODEL_PATH="THUDM/CogVideoX1.5-5b"    # HF 模型 ID 或本地路径
                                       # t2v 模型: CogVideoX-2b / CogVideoX-5b / CogVideoX1.5-5b
                                       # i2v 模型: CogVideoX-5b-I2V / CogVideoX1.5-5b-I2V
LORA_PATH=""                          # (可选) LoRA 权重目录路径

# === 生成任务配置 ===
GENERATE_TYPE="t2v"                   # 任务类型 (必须与模型匹配):
                                       # - t2v: 文本生成视频 (需要 t2v 模型)
                                       # - i2v: 图片生成视频 (需要 I2V 模型 + IMAGE_OR_VIDEO_PATH)
                                       # - v2v: 视频生成视频 (用 t2v 模型 + IMAGE_OR_VIDEO_PATH)

PROMPT="A serene sunrise over a mountain lake, a superman ruin the earth"  # 文本提示词

IMAGE_OR_VIDEO_PATH=""               # 输入文件路径:
                                       # - i2v: 必须提供图片路径 (如 /path/to/image.jpg)
                                       # - v2v: 必须提供视频路径 (如 /path/to/video.mp4)
                                       # - t2v: 留空

OUTPUT_PATH="./outputs/inference_${SLURM_JOB_ID}.mp4"  # 输出视频路径

# === 生成参数 ===
NUM_FRAMES=81                        # 生成帧数:
                                       # - CogVideoX 1.0 (2b/5b): 49 帧 (6秒@8fps)
                                       # - CogVideoX 1.5: 81 帧 (5秒@16fps) 或 161 帧 (10秒@16fps)

FPS=16                               # 视频帧率:
                                       # - CogVideoX 1.0: 8 fps
                                       # - CogVideoX 1.5: 16 fps

NUM_STEPS=50                         # 推理步数 (30-100, 越大质量越好但越慢)
GUIDANCE_SCALE=6.0                   # CFG 引导强度 (5.0-10.0, 控制与 prompt 的贴合度)

DTYPE="bfloat16"                     # 计算精度:
                                       # - float16: 推荐用于 CogVideoX-2b
                                       # - bfloat16: 推荐用于 CogVideoX-5b 和 1.5 系列

# (Optional) path to python environment activation script
CONDA_ACTIVATE_CMD="source activate CogVideoX" # or use your environment activation command

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
