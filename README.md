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

| 阶段 | 内容 |
|------|------|
| **阶段 0** | 下载项目代码 |
| **阶段 1** | 环境配置与依赖安装 |
| **阶段 2** | 下载模型权重 |
| **阶段 3** | 修改配置文件 |
| **阶段 4** | 准备数据集 |
| **阶段 5** | 运行验证 |

---

## 阶段 0: 下载项目模型 (Download Project Model)

### 从 HuggingFace 下载

```powershell
# 安装 git-lfs（如果尚未安装）
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# 克隆仓库（包含代码但不包含大文件）
hf download RachelHGF/Mirage-in-the-Eyes
```

> [!IMPORTANT]
> 克隆后请检查项目结构是否完整，确保包含以下关键目录：
> - `minigpt4/` - 核心代码
> - `eval/` - 评估脚本
> - `eval_configs/` - 配置文件
> - `transformers-4.29.2/` - 修改版 transformers
> - `attack.py` - 攻击脚本
> - `generate.py` - 生成脚本
> - `requirements.txt` - 依赖文件

---

## 阶段 1: 环境配置与依赖安装

### 1.1 创建 Conda 环境

```powershell
# 创建 Python 3.9 环境
conda create -n mllm python=3.9.20
conda activate mllm
```

### 1.2 安装依赖

```powershell
# 安装依赖
pip install -r requirements.txt
```

> [!IMPORTANT]
> 由于原项目可能是在服务器环境上运行，所以我们修改了依赖文件，适用于win11操作系统

```powershell
# 安装依赖
pip install -r requirements_windows.txt

# 安装修改版 transformers
python -m pip install -e transformers-4.29.2
```

### 1.3 补装额外依赖

```powershell
conda activate mllm
pip install sentencepiece accelerate peft timm einops open_clip_torch opencv-python omegaconf webdataset matplotlib pandas
```

### 1.4 验证安装

我们写了验证脚本，用于检查环境是否配置正确

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
