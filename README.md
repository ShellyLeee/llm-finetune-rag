# llm-finetune-rag

工业可复现的 LLM 微调 + 轻量 RAG + 幻觉率评测仓库骨架，默认面向 `Qwen/Qwen2.5-7B-Instruct`，训练后端复用服务器上已安装的 LLaMA-Factory，本仓库不修改其源码。

## 目标

- SFT LoRA / QLoRA 微调
- 轻量 RAG 检索与生成占位链路
- hallucination-aware 评测骨架，便于后续接入 citation-based / entailment-based 指标
- 可复现训练：配置文件版本化、输出集中写入 `runs/`、评测输出写入 `reports/`

## 环境要求

- Python >= 3.11
- CUDA 可用
- 目标服务器支持多 GPU 训练，推荐单机 `8x RTX 4090`
- 已在服务器侧单独安装好 LLaMA-Factory

## 一次性安装

1. 先在服务器上安装 LLaMA-Factory，例如放在 `~/llm_project/LlamaFactory`
2. 在你的训练环境中安装本仓库依赖

```bash
chmod +x scripts/*.sh
./scripts/install_deps.sh
```

3. 同时安装 LLaMA-Factory 侧需要的依赖，尤其是其 `requirements/metrics.txt` 与 `requirements/deepspeed.txt`

```bash
cd ~/llm_project/LlamaFactory
pip install -r requirements/metrics.txt
pip install -r requirements/deepspeed.txt
```

4. 做环境检查

```bash
./scripts/env_check.sh
```

## 仓库使用方式

### 1. 数据准备

输入数据至少包含 `instruction` 和 `output` 字段，可选 `lang`。本仓库当前导出为 `messages` 格式 JSONL，便于对接大多数 chat-style SFT 数据集。

示例输入 JSONL:

```json
{"instruction": "介绍一下监督微调。", "output": "监督微调是指...", "lang": "zh"}
```

执行转换:

```bash
python src/prepare_dataset.py \
  --input data/raw/your_dataset.jsonl \
  --output data/processed/train.jsonl \
  --dataset_name demo_sft \
  --lang_field lang
```

脚本会：

- 生成 `data/processed/train.jsonl`
- 生成 `data/processed/data_stats.json`
- 在输入缺失时自动创建占位流程提示，不会直接崩溃

`data/dataset_info.json` 已提供最小可用的 `demo_sft` 配置，默认指向 `data/processed/train.jsonl`。替换真实数据时，修改该文件中的路径与数据集名称即可。

如果 LLaMA-Factory 要求固定位置的 `dataset_info.json`，可在其 `data/` 目录中创建软链接：

```bash
ln -sf /Users/shellyli/Documents/GitHub/llm-finetune-rag/data/dataset_info.json \
  ~/llm_project/LlamaFactory/data/dataset_info.json
```

### 2. 训练

DeepSpeed 多卡训练脚本默认使用：

- 配置：`configs/train/sft_lora_qwen2.5_7b.yaml`
- ZeRO：`ds_config/zero2.json`
- GPU：若未设置 `CUDA_VISIBLE_DEVICES`，默认尝试 `0,1,2,3,4,5,6,7`

示例：

```bash
export LLAMA_FACTORY_DIR=~/llm_project/LlamaFactory
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/train_sft_deepspeed.sh configs/train/sft_lora_qwen2.5_7b.yaml ds_config/zero2.json
```

切换 ZeRO Stage：

```bash
./scripts/train_sft_deepspeed.sh configs/train/sft_lora_qwen2.5_7b.yaml ds_config/zero3.json
```

Torchrun 版本：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/train_sft_torchrun.sh configs/train/sft_lora_qwen2.5_7b.yaml
```

训练输出统一写入 `runs/`。脚本会尝试在对应输出目录下记录 `meta.txt`，写入 config 路径与当前 git commit。

### 3. 评测

```bash
./scripts/eval.sh
```

默认行为：

- 从 `data/eval` 读取样本
- 如果没有评测数据，自动创建少量 dummy 样本并提示
- 输出 `reports/results.jsonl` 和 `reports/summary.json`

### 4. 推理 / 部署

`scripts/serve_vllm.sh` 当前为占位脚本，用于：

- 检查 `vllm` 是否已安装
- 提示如何启动 OpenAI-compatible server
- 后续可扩展到 LoRA merge / base model serving

## 配置组织

- `configs/models/`: 模型级配置，可扩展到 Llama3 / Mistral / Qwen3.5 等
- `configs/train/`: 训练配置，当前含 LoRA 与 QLoRA 骨架
- `configs/inference/`: 推理配置占位
- `configs/export/`: LoRA merge 配置占位
- `ds_config/`: DeepSpeed ZeRO-2 / ZeRO-3

## 可复现性说明

- 所有训练关键配置均放在 `configs/` 并可版本化
- 训练产物统一写入 `runs/`
- `runs/` 与大文件目录已加入 `.gitignore`
- 数据处理产出固定写入 `data/processed/`
- 评测结果固定写入 `reports/`
- `seed` 已在训练配置中固定为 `42`
- TODO: 在训练前自动记录 `pip freeze`、CUDA/driver、LLaMA-Factory commit、DeepSpeed 版本
- TODO: 在 `meta.txt` 中补充完整日志路径、随机种子、数据版本与评测版本

## 目录

```text
.
├── configs/
├── data/
├── ds_config/
├── reports/
├── runs/
├── scripts/
└── src/
```

