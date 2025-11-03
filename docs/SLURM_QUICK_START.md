# CogVideo SLURM Quick Start（中文）

本说明配合 `scripts/slurm_train.sh` 与 `scripts/slurm_inference.sh` 使用，帮助你把任务通过 `sbatch` 提交到 SLURM GPU 集群。

目录
- 修改点快速说明
- 如何提交
- 示例：单节点 4GPU 训练
- 示例：单 GPU 推理
- 常见调整项说明

---

## 修改点快速说明
脚本顶部有一段 SBATCH 注释和一组 "USER CONFIGURABLE VARIABLES"（变量）。你通常需要按下列项修改：

- partition（分区名）: SBATCH `--partition`。
- gres/gpu（GPU 数）: SBATCH `--gres=gpu:N`（slurm 配置不同，可能需要 `--gpus`）。
- nodes（节点数）: 多节点训练时设置为 >1。
- cpus-per-task: 给数据加载/预处理保留的 CPU 数。
- mem / mem-per-cpu: 内存要求。
- time: 最大运行时。
- CONDA_ACTIVATE_CMD: 集群中激活 python 环境的命令（例：`source /etc/profile.d/conda.sh; conda activate cogvideo` 或模块加载 `module load anaconda`）。
- TRAIN_SCRIPT / TRAIN_ARGS（训练脚本位置）或 MODEL_PATH / PROMPT / OUTPUT_PATH（推理脚本）

请在提交前确保日志目录存在（脚本会创建 `logs/` 与 `outputs/`）。

---

## 如何提交
在仓库根目录下：

提交训练：
```bash
sbatch scripts/slurm_train.sh
```

提交推理：
```bash
sbatch scripts/slurm_inference.sh
```

查看日志：
```bash
less logs/train-<jobid>.out
less logs/inference-<jobid>.out
```

取消作业：
```bash
scancel <jobid>
```

---

## 示例：单节点 4GPU 训练（简短示例）
在 `scripts/slurm_train.sh` 中设置：
- `GPUS_PER_NODE=4`
- `NNODES=1`
- `TRAIN_SCRIPT=finetune/train.py` 或使用仓库内 `finetune/train_ddp_t2v.sh`

脚本的 `LAUNCH_CMD` 默认使用 `python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE}`。
如果你使用 DeepSpeed，请用 `deepspeed` 命令（并设置 `--deepspeed_config`）。

注意：多节点训练需正确配置 `MASTER_ADDR`（脚本会在多节点模式下自动从 SLURM_NODELIST 取首节点作为 master）。

---

## 示例：单 GPU 推理
在 `scripts/slurm_inference.sh` 中设置：
- `--gres=gpu:1`、`--cpus-per-task=8`
- `MODEL_PATH`（例如 `THUDM/CogVideoX1.5-5b`）
- `PROMPT`、`OUTPUT_PATH` 等变量

提交：
```bash
sbatch scripts/slurm_inference.sh
```

脚本会运行 `python inference/cli_demo.py --prompt ...` 并把生成的视频保存到 `OUTPUT_PATH`。

---

## 常见调整项说明
- `--gres` / `--gpus`：不同集群 SLURM 配置可能使用 `--gres=gpu:N` 或 `--gpus=N`，请按你们集群要求修改脚本的 SBATCH 行。
- 多节点训练：把 `#SBATCH --nodes` 设为节点数，并在脚本中设置 `NNODES` 对应；确保通往 master 的网络端口 (`MASTER_PORT`) 在防火墙中可用。
- Python 环境：多数集群使用模块系统或 conda，请把 `CONDA_ACTIVATE_CMD` 改为所在集群的环境加载命令。
- Deepspeed：若使用 DeepSpeed，请在 `LAUNCH_CMD` 中替换为 `deepspeed` 命令并提供 deepspeed config 文件（也可使用 accelerate/deepspeed integration）。
- I/O：训练/推理涉及大文件（权重、缓存），请确保 `OUTPUT_DIR` 有足够的磁盘空间，并尽量使用本地 SSD（而不是共享网络盘），以提高性能。

---

如果你希望我根据你们集群的实际 SLURM 参数（例如 partition 名、是否使用 `--gpus` 而非 `--gres`、模块加载命令）定制脚本，请把这些信息发给我，我会替你调整并测试脚本内容。祝你在集群上运行顺利！
