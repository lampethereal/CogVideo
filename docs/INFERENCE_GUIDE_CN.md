# CogVideo 推理完整指南（中文）

本文档详细说明 CogVideo 项目的推理流程、SLURM 脚本参数与如何调整关键参数以适配不同推理任务。

---

## 目录
1. [推理流程梳理](#推理流程梳理)
2. [SLURM 推理脚本关键参数详解](#slurm-推理脚本关键参数详解)
3. [启动任务完整流程](#启动任务完整流程)
4. [关键参数调整指南](#关键参数调整指南)
5. [常见问题与调试](#常见问题与调试)

---

## 推理流程梳理

### 整体架构
CogVideo 项目提供两种推理实现：
1. **Diffusers 版本**（推荐，易用）：位于 `inference/cli_demo.py`，使用 HuggingFace diffusers 库
2. **SAT 版本**（高性能）：位于 `sat/sample_video.py`，使用 SwissArmyTransformer

我们的 SLURM 脚本使用的是 **Diffusers 版本**。

### 推理核心步骤（基于 `inference/cli_demo.py`）

```
用户输入 (prompt + 配置参数)
    ↓
1. 根据 generate_type 选择 Pipeline
   - t2v (文本生成视频): CogVideoXPipeline
   - i2v (图片生成视频): CogVideoXImageToVideoPipeline
   - v2v (视频生成视频): CogVideoXVideoToVideoPipeline
    ↓
2. 从 HuggingFace 或本地加载预训练模型
   - 加载 transformer、VAE、text_encoder、scheduler
   - 根据 dtype (bfloat16/float16) 设置精度
    ↓
3. (可选) 加载 LoRA 权重并融合到模型
    ↓
4. 配置 Scheduler (DPM 或 DDIM)
    ↓
5. 启用内存优化
   - enable_sequential_cpu_offload() (串行 CPU 卸载)
   - VAE slicing 和 tiling (减少显存峰值)
    ↓
6. 根据 generate_type 准备输入
   - t2v: 仅需 prompt
   - i2v: prompt + image (从 image_or_video_path 加载)
   - v2v: prompt + video (从 image_or_video_path 加载)
    ↓
7. 调用 Pipeline 生成视频
   - 参数: prompt, num_frames, guidance_scale, num_inference_steps, seed 等
   - 使用 use_dynamic_cfg=True (DPM scheduler)
    ↓
8. 导出视频到指定路径 (MP4 格式)
```

### 关键代码位置
- **Pipeline 选择**: `cli_demo.py` L119-L125
- **模型加载**: `cli_demo.py` L119-L125 (`from_pretrained`)
- **LoRA 加载**: `cli_demo.py` L128-L132
- **生成调用**: `cli_demo.py` L156-L194
- **分辨率映射**: `cli_demo.py` L38-L48 (`RESOLUTION_MAP`)

---

## SLURM 推理脚本关键参数详解

脚本路径: `scripts/slurm_inference.sh`

### 1. SLURM 资源配置（#SBATCH 开头）

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `--job-name` | cogvideo_infer | 作业名称 | 可改为描述性名称如 `cogvideo_t2v_test` |
| `--partition` | gpu | 分区/队列名 | **必改**: 改为你集群的 GPU 分区名 |
| `--gres` | gpu:1 | GPU 数量 | 单 GPU 推理保持 `gpu:1`，多 GPU 改为 `gpu:N` |
| `--nodes` | 1 | 节点数 | 推理通常用 1 个节点 |
| `--ntasks-per-node` | 1 | 每节点任务数 | 保持 1 |
| `--cpus-per-task` | 8 | CPU 核心数 | 数据加载用，可根据集群调整 4-16 |
| `--mem` | 64G | 内存 | CogVideoX-5B 推荐 64G+，2B 可降到 32G |
| `--time` | 04:00:00 | 最大运行时间 | 根据生成视频数量调整 |
| `--output/--error` | logs/inference-%j.out | 日志路径 | `%j` 会被替换为 job ID |

### 2. 模型与数据路径参数

| 参数 | 默认值 | 说明 | 调整方法 |
|------|--------|------|----------|
| `MODEL_PATH` | THUDM/CogVideoX1.5-5b | 模型路径 | HF 模型 ID 或本地路径 |
| `LORA_PATH` | "" (空) | LoRA 权重路径 | 微调后填写 LoRA 目录路径 |
| `PROMPT` | "A serene sunrise..." | 文本提示词 | **必改**: 改为你要生成的内容 |
| `OUTPUT_PATH` | ./outputs/inference_${SLURM_JOB_ID}.mp4 | 输出视频路径 | 确保 outputs/ 目录存在 |
| `IMAGE_OR_VIDEO_PATH` | "" (空) | 输入图片/视频 | i2v/v2v 时**必填** |

### 3. 生成配置参数

| 参数 | 默认值 | 说明 | 调整影响 |
|------|--------|------|----------|
| `GENERATE_TYPE` | t2v | 生成类型 | **关键**: t2v / i2v / v2v (见下文详解) |
| `NUM_FRAMES` | 81 | 生成帧数 | 见模型兼容性表 |
| `NUM_STEPS` | 50 | 推理步数 | 越大质量越好但越慢，范围 30-100 |
| `GUIDANCE_SCALE` | 6.0 | CFG 引导强度 | 控制与 prompt 的贴合度，5-10 |
| `DTYPE` | bfloat16 | 精度类型 | 2B 用 float16，5B 用 bfloat16 |
| `FPS` | 16 | 视频帧率 | CogVideoX1.5 用 16，CogVideoX1.0 用 8 |

### 4. 环境配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CONDA_ACTIVATE_CMD` | source /path/to/conda.sh; conda activate cogvideo | 环境激活命令 |

**必须修改**为你集群的实际命令，例如:
```bash
# 如果使用 module 系统
CONDA_ACTIVATE_CMD="module load anaconda3; source activate cogvideo"

# 如果使用标准 conda
CONDA_ACTIVATE_CMD="source ~/miniconda3/etc/profile.d/conda.sh; conda activate cogvideo"
```

---

## 启动任务完整流程

### 步骤 1: 准备环境（首次使用）

```bash
# 在登录节点执行

# 1. 创建 conda 环境
conda create -n cogvideo python=3.11 -y
conda activate cogvideo

# 2. 安装依赖
cd /path/to/CogVideo
pip install -r requirements.txt

# 3. (可选) 预下载模型到本地，避免计算节点联网问题
python -c "from diffusers import CogVideoXPipeline; CogVideoXPipeline.from_pretrained('THUDM/CogVideoX1.5-5b')"
```

### 步骤 2: 修改 SLURM 脚本

```bash
# 编辑脚本
vim scripts/slurm_inference.sh

# 必须修改的项：
# 1. SBATCH --partition=<你的GPU分区>
# 2. CONDA_ACTIVATE_CMD=<你的环境激活命令>
# 3. PROMPT=<你的提示词>
# 4. 根据需要调整 GENERATE_TYPE、NUM_FRAMES 等
```

### 步骤 3: 创建输出目录

```bash
mkdir -p logs outputs
```

### 步骤 4: 提交作业

```bash
sbatch scripts/slurm_inference.sh
```

### 步骤 5: 监控作业

```bash
# 查看作业状态
squeue -u $USER

# 实时查看日志
tail -f logs/inference-<jobid>.out

# 作业完成后检查输出
ls -lh outputs/inference-<jobid>.mp4
```

---

## 关键参数调整指南

### 核心参数: GENERATE_TYPE

这是**最关键**的参数，决定推理任务类型。

#### 1. `GENERATE_TYPE="t2v"` (文本生成视频)

**用途**: 仅从文本描述生成视频

**必需参数**:
- `PROMPT`: 文本描述
- `MODEL_PATH`: 必须是 t2v 模型

**可选参数**:
- `NUM_FRAMES`, `NUM_STEPS`, `GUIDANCE_SCALE`

**兼容模型**:
- `THUDM/CogVideoX-2b`
- `THUDM/CogVideoX-5b`
- `THUDM/CogVideoX1.5-5b`

**脚本调整**:
```bash
# 在 slurm_inference.sh 中
GENERATE_TYPE="t2v"
MODEL_PATH="THUDM/CogVideoX1.5-5b"
PROMPT="A cat playing piano in a jazz club"
IMAGE_OR_VIDEO_PATH=""  # 保持为空
NUM_FRAMES=81           # CogVideoX1.5 用 81
FPS=16
```

**注意事项**:
- 无需提供 `IMAGE_OR_VIDEO_PATH`
- 脚本会自动跳过 `--image_or_video_path` 参数

---

#### 2. `GENERATE_TYPE="i2v"` (图片生成视频)

**用途**: 从静态图片生成视频（图片作为首帧或引导）

**必需参数**:
- `PROMPT`: 描述视频内容的文本
- `IMAGE_OR_VIDEO_PATH`: **必须提供**图片路径（.jpg/.png）
- `MODEL_PATH`: 必须是 i2v 模型

**兼容模型**:
- `THUDM/CogVideoX-5b-I2V`
- `THUDM/CogVideoX1.5-5b-I2V`

**脚本调整**:
```bash
# 在 slurm_inference.sh 中
GENERATE_TYPE="i2v"
MODEL_PATH="THUDM/CogVideoX1.5-5b-I2V"  # 注意模型名必须包含 I2V
PROMPT="The cat starts moving and playing with a toy"
IMAGE_OR_VIDEO_PATH="/path/to/your/image.jpg"  # 必填
NUM_FRAMES=81
FPS=16
```

**注意事项**:
1. **模型路径必须改**: t2v 模型不能用于 i2v，必须用专门的 I2V 模型
2. **图片必须存在**: 脚本会检查 `IMAGE_OR_VIDEO_PATH` 非空才传参
3. **分辨率处理**: CogVideoX1.5-I2V 支持任意分辨率，会自动调整

---

#### 3. `GENERATE_TYPE="v2v"` (视频生成视频)

**用途**: 基于已有视频生成新视频（风格转换、内容变化）

**必需参数**:
- `PROMPT`: 描述目标视频内容
- `IMAGE_OR_VIDEO_PATH`: **必须提供**输入视频路径（.mp4/.avi）
- `MODEL_PATH`: 使用 t2v 模型（v2v 复用 t2v 模型）

**兼容模型**:
- `THUDM/CogVideoX-2b`
- `THUDM/CogVideoX-5b`
- `THUDM/CogVideoX1.5-5b`

**脚本调整**:
```bash
# 在 slurm_inference.sh 中
GENERATE_TYPE="v2v"
MODEL_PATH="THUDM/CogVideoX1.5-5b"  # 使用 t2v 模型
PROMPT="Transform the scene into a watercolor painting style"
IMAGE_OR_VIDEO_PATH="/path/to/input_video.mp4"  # 必填
NUM_FRAMES=81
FPS=16
```

**注意事项**:
1. **输入视频帧数**: 会自动截取前 N 帧（N = NUM_FRAMES），建议输入视频足够长
2. **v2v 用 t2v 模型**: 不需要专门的 v2v 模型，用 t2v 模型即可
3. **提示词作用**: prompt 引导视频转换方向

---

### 参数联动关系

#### 1. MODEL_PATH ↔ GENERATE_TYPE ↔ NUM_FRAMES ↔ FPS

| 模型系列 | 支持类型 | 推荐帧数 | FPS | 分辨率 (H×W) |
|----------|----------|----------|-----|--------------|
| CogVideoX-2b | t2v, v2v | 49 | 8 | 480×720 |
| CogVideoX-5b | t2v, v2v | 49 | 8 | 480×720 |
| CogVideoX-5b-I2V | i2v | 49 | 8 | 480×720 |
| CogVideoX1.5-5b | t2v, v2v | 81 | 16 | 768×1360 |
| CogVideoX1.5-5b-I2V | i2v | 81 | 16 | 任意(建议768×1360) |

**调整规则**:
```bash
# CogVideoX 1.0 系列 (2B/5B)
if [[ "$MODEL_PATH" == *"CogVideoX-"* ]] && [[ "$MODEL_PATH" != *"1.5"* ]]; then
  NUM_FRAMES=49
  FPS=8
fi

# CogVideoX 1.5 系列
if [[ "$MODEL_PATH" == *"CogVideoX1.5"* ]]; then
  NUM_FRAMES=81
  FPS=16
fi

# I2V 模型必须用 i2v 类型
if [[ "$MODEL_PATH" == *"I2V"* ]]; then
  GENERATE_TYPE="i2v"
  IMAGE_OR_VIDEO_PATH="/path/to/image.jpg"  # 必填
fi
```

#### 2. DTYPE ↔ MODEL_PATH

```bash
# CogVideoX-2B 推荐用 float16
if [[ "$MODEL_PATH" == *"2b"* ]]; then
  DTYPE="float16"
fi

# CogVideoX-5B 和 1.5 系列推荐用 bfloat16
if [[ "$MODEL_PATH" == *"5b"* ]]; then
  DTYPE="bfloat16"
fi
```

#### 3. GPU 内存 ↔ 模型 ↔ 配置

| 模型 | 最小显存 | 推荐显存 | 优化建议 |
|------|----------|----------|----------|
| CogVideoX-2b | 18GB | 24GB | float16 + CPU offload |
| CogVideoX-5b | 26GB | 40GB | bfloat16 + CPU offload |
| CogVideoX1.5-5b | 40GB+ | 80GB | 启用所有优化 |

**SLURM 脚本调整**:
```bash
# 对于 2B 模型（低显存）
#SBATCH --gres=gpu:1          # RTX 3090/4090 (24GB)
#SBATCH --mem=32G

# 对于 5B 模型（中等显存）
#SBATCH --gres=gpu:1          # A100 (40GB/80GB)
#SBATCH --mem=64G

# 对于 1.5-5B 模型（高显存）
#SBATCH --gres=gpu:1          # A100 80GB
#SBATCH --mem=128G
```

---

### 完整参数调整检查清单

更改 `GENERATE_TYPE` 时，必须检查以下项：

#### 从 t2v 改为 i2v：
- [ ] 修改 `GENERATE_TYPE="i2v"`
- [ ] 修改 `MODEL_PATH` 为 I2V 模型（如 `THUDM/CogVideoX-5b-I2V`）
- [ ] 设置 `IMAGE_OR_VIDEO_PATH="/path/to/image.jpg"`
- [ ] 确认图片文件存在且可访问
- [ ] 检查 `NUM_FRAMES`、`FPS` 是否匹配模型
- [ ] （可选）调整 `--mem` 和 `--time`

#### 从 t2v 改为 v2v：
- [ ] 修改 `GENERATE_TYPE="v2v"`
- [ ] 保持 `MODEL_PATH` 为 t2v 模型（**不改**）
- [ ] 设置 `IMAGE_OR_VIDEO_PATH="/path/to/video.mp4"`
- [ ] 确认输入视频存在且帧数充足
- [ ] 修改 `PROMPT` 描述转换目标
- [ ] （可选）增加 `--time`（v2v 通常更慢）

#### 从 i2v 改为 t2v：
- [ ] 修改 `GENERATE_TYPE="t2v"`
- [ ] 修改 `MODEL_PATH` 为 t2v 模型（去掉 `-I2V` 后缀）
- [ ] 清空 `IMAGE_OR_VIDEO_PATH=""`
- [ ] 检查 `NUM_FRAMES`、`FPS` 是否匹配新模型

---

## 常见问题与调试

### 问题 1: 作业提交后立即失败

**症状**: `squeue` 中看不到作业或立即进入 FAILED 状态

**排查步骤**:
```bash
# 查看错误日志
cat logs/inference-<jobid>.err

# 常见原因:
# 1. 分区名错误
sinfo  # 查看可用分区

# 2. GPU 资源不足
squeue -p gpu  # 查看队列

# 3. 环境激活失败
# 手动测试激活命令是否正确
```

**解决方案**:
```bash
# 在脚本中修改
#SBATCH --partition=<正确的分区名>

# 或在提交时覆盖
sbatch --partition=correct_gpu_partition scripts/slurm_inference.sh
```

### 问题 2: CUDA out of memory

**症状**: 日志显示 `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 1. 降低分辨率（在 cli_demo.py 中会自动处理）
# 2. 减少帧数
NUM_FRAMES=49  # 从 81 降到 49

# 3. 启用更多内存优化（已默认开启）
# cli_demo.py 中已包含:
# - enable_sequential_cpu_offload()
# - vae.enable_slicing()
# - vae.enable_tiling()

# 4. 使用量化（需修改代码调用 cli_demo_quantization.py）
```

### 问题 3: i2v 类型找不到图片

**症状**: `FileNotFoundError: image not found`

**检查**:
```bash
# 1. 确认路径正确（绝对路径更保险）
IMAGE_OR_VIDEO_PATH="/absolute/path/to/image.jpg"

# 2. 确认文件在计算节点可访问
# 如果图片在网络盘，确保挂载正确

# 3. 检查文件权限
ls -l /path/to/image.jpg
```

### 问题 4: 模型下载失败

**症状**: `Connection timeout` 或 `403 Forbidden`

**解决方案**:
```bash
# 方案1: 预下载到本地（推荐）
# 在登录节点执行
python << EOF
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX1.5-5b")
pipe.save_pretrained("/path/to/local/model")
EOF

# 然后修改脚本
MODEL_PATH="/path/to/local/model"

# 方案2: 配置 HF 镜像（如果在国内）
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 5: 生成的视频质量差

**调优建议**:
```bash
# 1. 增加推理步数
NUM_STEPS=100  # 从 50 提高到 100

# 2. 调整 guidance scale
GUIDANCE_SCALE=7.0  # 尝试 6-10 范围

# 3. 优化提示词
# 使用 inference/convert_demo.py 优化 prompt
python inference/convert_demo.py --prompt "简短描述" --type t2v

# 4. 使用更大的模型
MODEL_PATH="THUDM/CogVideoX1.5-5b"  # 从 2b 升级到 5b/1.5
```

---

## 快速参考

### 最小可运行配置（t2v）

```bash
# slurm_inference.sh 关键行
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

MODEL_PATH="THUDM/CogVideoX1.5-5b"
GENERATE_TYPE="t2v"
PROMPT="Your prompt here"
IMAGE_OR_VIDEO_PATH=""
NUM_FRAMES=81
FPS=16
DTYPE="bfloat16"
```

### i2v 完整示例

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

MODEL_PATH="THUDM/CogVideoX1.5-5b-I2V"
GENERATE_TYPE="i2v"
PROMPT="The dog starts running in the park"
IMAGE_OR_VIDEO_PATH="/data/images/dog.jpg"
NUM_FRAMES=81
FPS=16
DTYPE="bfloat16"
```

### v2v 完整示例

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00

MODEL_PATH="THUDM/CogVideoX1.5-5b"
GENERATE_TYPE="v2v"
PROMPT="Transform into anime style with vibrant colors"
IMAGE_OR_VIDEO_PATH="/data/videos/input.mp4"
NUM_FRAMES=81
FPS=16
DTYPE="bfloat16"
```

---

## 批量提交技巧

### 方法 1: 使用循环（多个 prompt）

```bash
# 创建 prompts.txt
cat > prompts.txt << EOF
A cat playing piano
A dog running in the park
A bird flying over mountains
EOF

# 批量提交
while IFS= read -r prompt; do
  sbatch --export=ALL,PROMPT="$prompt" scripts/slurm_inference.sh
done < prompts.txt
```

### 方法 2: 修改脚本支持参数

在 `slurm_inference.sh` 开头添加:
```bash
# 支持命令行参数
PROMPT="${1:-A serene sunrise over a mountain lake}"
MODEL_PATH="${2:-THUDM/CogVideoX1.5-5b}"
```

提交:
```bash
sbatch scripts/slurm_inference.sh "Custom prompt" "THUDM/CogVideoX-5b"
```

---

如有其他问题，请查看:
- 项目主 README: `README.md`
- SLURM 快速开始: `docs/SLURM_QUICK_START.md`
- 推理脚本源码: `inference/cli_demo.py`
