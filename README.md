---
license: mit
---
# _Mirage-in-the-Eyes_: Hallucination Attack on Multi-modal Large Language Models with _Only_ Attention Sink

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

**NOTE**: To prevent potential harm, we release our source code only *upon request for research purposes*.

## 📖 项目概述 (Project Overview)

**Mirage-in-the-Eyes** 是一个针对多模态大语言模型 (MLLMs) 的**幻觉攻击**研究项目。

本项目深入研究 MLLMs 的内部注意力机制，揭示幻觉问题的根本原因，暴露指令微调过程中的固有漏洞。我们提出了一种新颖的幻觉攻击方法，利用**注意力沉降**行为触发与图像-文本相关性最小的幻觉内容，对关键下游应用构成重大威胁。

与以往依赖固定模式的对抗方法不同，我们的方法生成动态、有效且高度可迁移的视觉对抗输入，而不会牺牲模型响应的质量。在 6 个知名 MLLMs 上的广泛实验证明了我们的攻击在破坏具有广泛防御机制的黑盒 MLLMs 方面的有效性，以及对 GPT-4o 和 Gemini 1.5 等最先进商业 API 的良好效果。

> [!NOTE]
> **复现范围**：由于 RTX 4060 (8GB) 显存限制，本指南仅复现 **MiniGPT-4 (Vicuna-7B)** 部分。

---

## 🚀 快速开始 (Quick Start)

### 总体步骤

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| **阶段 0** | 下载项目代码 | 5 分钟 |
| **阶段 1** | 环境配置与依赖安装 | 10-15 分钟 |
| **阶段 2** | 下载模型权重 | 1-2 小时 |
| **阶段 3** | 修改配置文件 | 10 分钟 |
| **阶段 4** | 准备数据集 | 30-60 分钟 |
| **阶段 5** | 运行验证 | 15-30 分钟 |
| **总计** | - | **约 2-4 小时** |

---

## 阶段 0: 下载项目代码 (Download Project Code)

### 方式 1: 从 HuggingFace 下载（推荐）

```powershell
# 安装 git-lfs（如果尚未安装）
# 下载地址: https://git-lfs.github.com/
git lfs install

# 克隆仓库（包含代码但不包含大文件）
git clone https://huggingface.co/spaces/YOUR_USERNAME/Mirage-in-the-Eyes
cd Mirage-in-the-Eyes

# 或者使用 HuggingFace CLI 下载
pip install huggingface_hub
huggingface-cli download YOUR_USERNAME/Mirage-in-the-Eyes --repo-type space --local-dir ./Mirage-in-the-Eyes
```

### 方式 2: 从其他源下载

```powershell
# 如果从其他 Git 仓库克隆
git clone <your-repo-url>
cd Mirage-in-the-Eyes
```

> [!IMPORTANT]
> 克隆后请检查项目结构是否完整，确保包含以下关键目录：
> - `minigpt4/` - 核心代码
> - `eval/` - 评估脚本
> - `eval_configs/` - 配置文件
> - `transformers-4.29.2/` - 修改版 transformers

---

## 阶段 1: 环境配置与依赖安装 (Environment Setup)

### 1.1 创建 Conda 环境

```powershell
# 创建 Python 3.9 环境
conda create -n mllm python=3.9.20
conda activate mllm
```

### 1.2 安装基础依赖

**Windows 用户：**

```powershell
# 安装基础依赖
pip install -r requirements_windows.txt

# 安装修改版 transformers
python -m pip install -e transformers-4.29.2
```

**Linux/Mac 用户：**

```powershell
# 安装基础依赖
pip install -r requirements.txt

# 安装修改版 transformers
python -m pip install -e transformers-4.29.2
```

### 1.3 补装额外依赖

```powershell
conda activate mllm
pip install sentencepiece accelerate peft timm einops open_clip_torch opencv-python omegaconf webdataset matplotlib pandas
```

### 1.4 验证安装

```powershell
python verify_install.py
```

> [!TIP]
> 如果出现依赖错误，请检查 CUDA 和 PyTorch 版本是否兼容。推荐：
> - CUDA 11.7+
> - PyTorch 2.0+

---

## 阶段 2: 下载模型权重 (Download Model Weights)

### 2.1 Vicuna-7B-v0 (~13GB)

#### 方式 1: 使用 HuggingFace CLI（推荐）

```powershell
# 创建目录
mkdir -p "D:\AI PROJEAT\mirage\weights\vicuna"

# 登录 Hugging Face（首次使用需要）
huggingface-cli login
# 访问 https://huggingface.co/settings/tokens 获取 token

# 下载模型
huggingface-cli download lmsys/vicuna-7b-v0 --local-dir "D:\AI PROJEAT\mirage\weights\vicuna\vicuna-7b-v0"
```

#### 方式 2: 手动下载

1. 访问：https://huggingface.co/lmsys/vicuna-7b-v0
2. 点击 "Files and versions" 标签
3. 下载所有文件到 `D:\AI PROJEAT\mirage\weights\vicuna\vicuna-7b-v0\`

### 2.2 MiniGPT-4 预训练权重 (~5GB)

#### 下载步骤

1. 访问 MiniGPT-4 官方仓库：https://github.com/Vision-CAIR/MiniGPT-4
2. 在 README 中找到预训练权重下载链接，或直接访问：
   - [pretrained_minigpt4_7b.pth](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
3. 下载后保存到：`D:\AI PROJEAT\mirage\weights\minigpt4\pretrained_minigpt4_7b.pth`

```powershell
# 创建目录
mkdir -p "D:\AI PROJEAT\mirage\weights\minigpt4"

# 使用 gdown 下载（需先安装: pip install gdown）
gdown https://drive.google.com/uc?id=1a4zLvaiDBr-36pasffmgpvH5P7CKmpze
move pretrained_minigpt4_7b.pth "D:\AI PROJEAT\mirage\weights\minigpt4\"
```

> [!WARNING]
> **显存限制提醒**
> - RTX 4060 (8GB) 仅支持 Vicuna-7B 模型
> - 不要下载 Vicuna-13B 或更大的模型
> - 确保使用 `vicuna-7b-v0` 而非 `vicuna-7b-v1.5`

---

## 阶段 3: 修改配置文件 (Configuration)

### 3.1 修改评估配置

编辑文件：`eval_configs/minigpt4_eval.yaml`

```yaml
# 第 8 行修改为（使用正斜杠 / 或双反斜杠 \\）:
ckpt: 'D:/AI PROJEAT/mirage/weights/minigpt4/pretrained_minigpt4_7b.pth'
```

### 3.2 修改模型配置

编辑文件：`minigpt4/configs/models/minigpt4_vicuna0.yaml`

```yaml
# 第 18 行修改为:
llama_model: "D:/AI PROJEAT/mirage/weights/vicuna/vicuna-7b-v0"
```

> [!TIP]
> 使用绝对路径可以避免路径错误。Windows 用户可以使用正斜杠 `/` 或双反斜杠 `\\`。

---

## 阶段 4: 准备数据集 (Data Preparation)

### 4.1 Hallubench 数据集

#### 下载方式

根据 [HA-DPO 官方仓库](https://github.com/opendatalab/HA-DPO) 提供的方法下载 Hallubench 数据集。

```powershell
# 克隆 HA-DPO 仓库获取下载脚本
git clone https://github.com/opendatalab/HA-DPO.git
cd HA-DPO

# 按照官方 README 说明下载数据集
# 通常包含：
# - hallubench.json (评测数据)
# - 对应的图片文件
```

#### 目录结构

```
D:\AI PROJEAT\mirage\data\
├── hallubench/
│   └── hallubench.json    # 数据集 JSON 文件
└── images/
    └── *.jpg              # 对应的图片
```

### 4.2 Visual Genome 数据集（可选）

如需训练或完整复现，请下载 Visual Genome 数据集：

**下载链接：**
- 图像数据集（Part 1）: [VG_100K](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) (~15GB)
- 图像数据集（Part 2）: [VG_100K_2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) (~15GB)
- 元数据: [image_data.json](http://visualgenome.org/static/data/dataset/image_data.json.zip) (~17MB)
- 区域描述: [region_descriptions.json](http://visualgenome.org/static/data/dataset/region_descriptions.json.zip) (~712MB)

**目录结构：**
```
D:\AI PROJEAT\mirage\data\VG\
├── VG_100K/                    # 64,346 张图片
├── VG_100K_2/                  # 43,903 张图片
├── image_data.json             # 图像元数据
└── region_descriptions.json    # 区域描述数据
```

---

## 阶段 5: 运行验证 (Running & Verification)

### 5.1 推理测试

```powershell
conda activate mllm
python generate.py `
  --model minigpt4 `
  --gpu-id 0 `
  --data-path "D:\AI PROJEAT\mirage\data\hallubench\hallubench.json" `
  --images-path "D:\AI PROJEAT\mirage\data\images" `
  --response-path "D:\AI PROJEAT\mirage\output\response.json" `
  --generation-mode greedy
```

### 5.2 幻觉攻击

```powershell
python attack.py `
  --model minigpt4 `
  --gpu-id 0 `
  --data-path "D:\AI PROJEAT\mirage\data\hallubench\hallubench.json" `
  --images-path "D:\AI PROJEAT\mirage\data\images" `
  --save-path "D:\AI PROJEAT\mirage\output\adv_images" `
  --generation-mode greedy `
  --eps 2
```

### 5.3 评估结果

```powershell
cd eval
python json_eval.py `
  --json-file "../output/response.json" `
  --bench-path "../data/hallubench" `
  --log-path "../output/log"
```

> [!TIP]
> **监控显存使用**
> ```powershell
> # 在另一个终端窗口运行
> nvidia-smi -l 1
> ```
> 确保显存占用不超过 8GB

---

## 📁 项目结构 (Project Structure)

```
Mirage-in-the-Eyes/
├── .gitignore                  # Git 忽略文件配置
├── README.md                   # 本文档
├── LICENSE                     # MIT 许可证
├── requirements.txt            # Python 依赖（Linux/Mac）
├── requirements_windows.txt    # Python 依赖（Windows）
├── attack.py                   # 主要攻击实现
├── generate.py                 # MLLM 响应生成
├── test_small_sample.py        # 小样本测试
├── verify_install.py           # 环境验证脚本
├── transformers-4.29.2/        # 修改版 transformers 库
├── minigpt4/                   # MiniGPT-4 核心代码
│   ├── configs/                # 模型配置
│   ├── models/                 # 模型定义
│   └── ...
├── eval/                       # 评估脚本
├── eval_configs/               # 评估配置
│   └── minigpt4_eval.yaml
├── data/                       # 数据集目录（需自行下载）
│   ├── hallubench/             # Hallubench 数据集
│   ├── images/                 # 图片文件
│   └── VG/                     # Visual Genome 数据集（可选）
├── weights/                    # 模型权重（需自行下载）
│   ├── minigpt4/               # MiniGPT-4 权重
│   │   └── pretrained_minigpt4_7b.pth
│   └── vicuna/                 # Vicuna 权重
│       └── vicuna-7b-v0/
└── output/                     # 输出目录
    ├── response.json           # 生成的响应
    ├── adv_images/             # 对抗样本图片
    └── log/                    # 评估日志
```

---

## ⚠️ 注意事项 (Important Notes)

### 硬件要求

> [!WARNING]
> **RTX 4060 (8GB) 显存限制**
> - 仅使用 Vicuna-7B 模型，避免 13B 或更大模型
> - 运行前关闭所有占用 GPU 的程序（浏览器、其他模型等）
> - 使用 `batch_size=1`
> - 如果出现 OOM (Out of Memory) 错误，尝试：
>   - 减小输入图片分辨率
>   - 使用 `torch.cuda.empty_cache()`
>   - 启用梯度检查点 (gradient checkpointing)

### 数据集要求

1. **Hallubench 数据集**: 必需，用于评测（~1-2GB）
2. **Visual Genome 数据集**: 可选，用于训练（~30GB）
3. **存储空间**: 确保至少有 50GB 可用空间（包括模型权重）

### 网络要求

1. **HuggingFace 访问**: 部分地区可能需要代理
2. **下载速度**: 模型权重约 18GB，建议使用稳定网络
3. **断点续传**: 使用 `huggingface-cli` 支持断点续传

### 环境兼容性

1. **CUDA 版本**: 确保 CUDA 11.7+ 
2. **PyTorch 版本**: 推荐 PyTorch 2.0+
3. **Python 版本**: 必须使用 Python 3.9.x
4. **操作系统**: Windows 10/11, Linux (Ubuntu 20.04+), macOS (Intel/Apple Silicon)

---

## 🔧 常见问题 (Troubleshooting)

### 1. HuggingFace 下载失败

```powershell
# 设置国内镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
```

### 2. 显存不足 (OOM)

```python
# 在代码中添加
import torch
torch.cuda.empty_cache()

# 或使用较小的批次大小
# 修改配置文件中的 batch_size 为 1
```

### 3. 依赖冲突

```powershell
# 重新创建环境
conda deactivate
conda env remove -n mllm
conda create -n mllm python=3.9.20
conda activate mllm

# 按顺序重新安装依赖
pip install -r requirements_windows.txt
python -m pip install -e transformers-4.29.2
pip install sentencepiece accelerate peft timm einops
```

### 4. 路径错误

确保所有路径使用：
- 绝对路径
- 正斜杠 `/` 或双反斜杠 `\\`
- 不包含中文字符

---

## 🙏 致谢 (Acknowledgement)

This repository is based on the MLLM codebase of [OPERA](https://github.com/shikiw/OPERA/tree/main). We sincerely thank the contributors for their valuable work.

Special thanks to:
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) for the foundational model
- [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) for the language model backbone
- [HA-DPO](https://github.com/opendatalab/HA-DPO) for the Hallubench dataset

---

## 📚 相关资源 (Related Resources)

- **论文**: [Link to paper] (待发布)
- **MiniGPT-4 官方文档**: https://minigpt-4.github.io/
- **Vicuna 模型**: https://lmsys.org/blog/2023-03-30-vicuna/
- **Hallubench 数据集**: https://github.com/opendatalab/HA-DPO
- **OPERA 项目**: https://github.com/shikiw/OPERA

---

## 📧 联系方式 (Contact)

如有问题或合作意向，请通过以下方式联系：

- **GitHub Issues**: [提交 Issue](https://github.com/YOUR_USERNAME/Mirage-in-the-Eyes/issues)
- **Email**: your.email@example.com
- **Pull Requests**: 欢迎贡献代码

---

## 📄 许可证 (License)

本项目采用 [MIT License](LICENSE)。

```
MIT License

Copyright (c) 2024 Mirage-in-the-Eyes Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🔖 版本历史 (Version History)

- **v1.0.0** (2024-12-23)
  - 初始发布
  - 支持 MiniGPT-4 (Vicuna-7B) 复现
  - RTX 4060 8GB 优化版本
