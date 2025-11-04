# CogVideo 推理快速决策卡 🚀

## 我该用哪个模型？

```
你的需求                           推荐模型                    GENERATE_TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
仅文本生成视频                 CogVideoX1.5-5b              t2v
质量优先，有80GB显存           CogVideoX1.5-5b              t2v
速度优先，有24GB显存           CogVideoX-2b                 t2v
平衡质量速度，有40GB显存       CogVideoX-5b                 t2v

从图片生成视频                 CogVideoX1.5-5b-I2V          i2v
图片生成，任意分辨率           CogVideoX1.5-5b-I2V          i2v
图片生成，标准分辨率           CogVideoX-5b-I2V             i2v

视频风格转换/内容变化          CogVideoX1.5-5b              v2v
视频转换，低显存               CogVideoX-2b                 v2v
```

## GENERATE_TYPE 决策树

```
开始
  ↓
有输入文件吗？
  ├─ 否 → 用 t2v (仅文本生成)
  │       MODEL_PATH = "THUDM/CogVideoX1.5-5b"
  │       IMAGE_OR_VIDEO_PATH = ""
  │
  └─ 是 → 是图片还是视频？
          ├─ 图片 (.jpg/.png) → 用 i2v (图片生成视频)
          │                    MODEL_PATH = "THUDM/CogVideoX1.5-5b-I2V"  ← 注意 I2V 后缀！
          │                    IMAGE_OR_VIDEO_PATH = "/path/to/image.jpg"
          │
          └─ 视频 (.mp4/.avi) → 用 v2v (视频转换)
                                MODEL_PATH = "THUDM/CogVideoX1.5-5b"  ← 用 t2v 模型！
                                IMAGE_OR_VIDEO_PATH = "/path/to/video.mp4"
```

## 参数联动速查表

| 改了这个          | 必须检查这些                                          |
|-------------------|-------------------------------------------------------|
| `GENERATE_TYPE`   | `MODEL_PATH`, `IMAGE_OR_VIDEO_PATH`, `NUM_FRAMES`, `FPS` |
| `MODEL_PATH`      | `DTYPE`, `NUM_FRAMES`, `FPS`, `--mem`, `--gres`      |
| `NUM_FRAMES`      | `FPS` (确保符合模型), `--time` (帧数多需更长时间)    |
| 模型从 2b→5b      | `DTYPE` (float16→bfloat16), `--mem` (32G→64G)       |
| 模型从 1.0→1.5    | `NUM_FRAMES` (49→81), `FPS` (8→16)                  |

## 三种类型完整配置对比

### 🎨 t2v (文本生成视频)

```bash
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

MODEL_PATH="THUDM/CogVideoX1.5-5b"
GENERATE_TYPE="t2v"
PROMPT="A beautiful sunset over the ocean"
IMAGE_OR_VIDEO_PATH=""           # ← 留空！
NUM_FRAMES=81
FPS=16
DTYPE="bfloat16"
```

**特点**: 最简单，只需要文本描述

---

### 🖼️ i2v (图片生成视频)

```bash
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

MODEL_PATH="THUDM/CogVideoX1.5-5b-I2V"    # ← 必须有 I2V！
GENERATE_TYPE="i2v"
PROMPT="The cat starts to walk forward"
IMAGE_OR_VIDEO_PATH="/data/cat.jpg"      # ← 必须填！
NUM_FRAMES=81
FPS=16
DTYPE="bfloat16"
```

**特点**: 以图片为起点，prompt 引导动作

**常见错误**:
- ❌ 用了 t2v 模型 (没有 I2V 后缀)
- ❌ IMAGE_OR_VIDEO_PATH 留空
- ❌ 提供了视频而不是图片

---

### 🎬 v2v (视频生成视频)

```bash
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00    # ← v2v 通常更慢

MODEL_PATH="THUDM/CogVideoX1.5-5b"        # ← 用 t2v 模型！
GENERATE_TYPE="v2v"
PROMPT="Transform to watercolor painting style"
IMAGE_OR_VIDEO_PATH="/data/input.mp4"    # ← 必须填视频！
NUM_FRAMES=81
FPS=16
DTYPE="bfloat16"
```

**特点**: 视频风格迁移，prompt 引导转换方向

**常见错误**:
- ❌ 用了 I2V 模型 (应该用 t2v 模型)
- ❌ IMAGE_OR_VIDEO_PATH 留空
- ❌ 输入视频帧数不足

---

## 显存不足救急方案

```bash
# 方案1: 降低帧数
NUM_FRAMES=49   # 从 81 降到 49

# 方案2: 用更小的模型
MODEL_PATH="THUDM/CogVideoX-2b"  # 从 5b 降到 2b
DTYPE="float16"

# 方案3: 增加交换内存 (SLURM)
#SBATCH --mem=128G

# 方案4: 使用量化（需改用 cli_demo_quantization.py）
```

## 常见错误速查

| 错误信息                              | 原因                | 解决方案                          |
|---------------------------------------|---------------------|-----------------------------------|
| `model_name not in RESOLUTION_MAP`    | 模型名不识别        | 检查 MODEL_PATH 拼写              |
| `image not found`                     | i2v 缺少图片        | 填写 IMAGE_OR_VIDEO_PATH          |
| `Expected 4D/5D tensor`               | 类型与模型不匹配    | i2v 必须用 I2V 模型               |
| `CUDA out of memory`                  | 显存不足            | 见上方救急方案                    |
| `Connection timeout` (下载)           | 网络问题            | 预下载模型到本地或配置镜像        |

## 性能调优速查

| 目标            | 调整参数                  | 建议值          |
|-----------------|---------------------------|-----------------|
| 提高质量        | NUM_STEPS                 | 50 → 100        |
| 提高质量        | GUIDANCE_SCALE            | 6.0 → 8.0       |
| 加快速度        | NUM_STEPS                 | 50 → 30         |
| 加快速度        | 用更小模型                | 5b → 2b         |
| 降低显存        | NUM_FRAMES                | 81 → 49         |
| 降低显存        | 启用 CPU offload (已默认) | -               |

## 提交前最后检查 ✅

```bash
# 1. 路径检查
[ -f "scripts/slurm_inference.sh" ] && echo "✅ 脚本存在"
[ -d "logs" ] || mkdir -p logs && echo "✅ 日志目录"
[ -d "outputs" ] || mkdir -p outputs && echo "✅ 输出目录"

# 2. 模型与类型匹配检查
if [[ "$GENERATE_TYPE" == "i2v" ]]; then
  [[ "$MODEL_PATH" == *"I2V"* ]] && echo "✅ i2v 用 I2V 模型" || echo "❌ i2v 必须用 I2V 模型"
  [[ -n "$IMAGE_OR_VIDEO_PATH" ]] && echo "✅ 已设置输入图片" || echo "❌ i2v 必须提供图片"
fi

if [[ "$GENERATE_TYPE" == "v2v" ]]; then
  [[ "$MODEL_PATH" != *"I2V"* ]] && echo "✅ v2v 用 t2v 模型" || echo "❌ v2v 不能用 I2V 模型"
  [[ -n "$IMAGE_OR_VIDEO_PATH" ]] && echo "✅ 已设置输入视频" || echo "❌ v2v 必须提供视频"
fi

# 3. 提交测试
sbatch --test-only scripts/slurm_inference.sh && echo "✅ 脚本语法正确"
```

## 一键测试命令

```bash
# t2v 测试
sbatch --job-name=test_t2v --time=00:30:00 scripts/slurm_inference.sh

# 查看状态
squeue -u $USER

# 取消测试作业
scancel $(squeue -u $USER -h -o "%i" | head -1)
```

---

**完整文档**: `docs/INFERENCE_GUIDE_CN.md`  
**SLURM 快速开始**: `docs/SLURM_QUICK_START.md`
