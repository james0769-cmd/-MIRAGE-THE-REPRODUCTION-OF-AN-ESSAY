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

### 工作流程

本项目的完整工作流程分为三个核心阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                    阶段 1: 对抗攻击                           │
│                   [attack.py]                                │
├─────────────────────────────────────────────────────────────┤
│  输入: 原始图片 (image.jpg)                                   │
│    ↓                                                         │
│  1. 读取原始图片                                              │
│  2. 生成对抗扰动 (利用注意力流程)                             │
│  3. 输出: 对抗图片 (adv_image.png)                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    阶段 2: 描述生成                           │
│                   [generate.py]                              │
├─────────────────────────────────────────────────────────────┤
│  路径 A (Baseline):                                          │
│    原始图片 → 模型推理 → 原始描述                             │
│                                                              │
│  路径 B (Hallucinated):                                      │
│    对抗图片 → 模型推理 → 对抗后描述                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    阶段 3: 效果评估                           │
├─────────────────────────────────────────────────────────────┤
│  对比分析:                                                    │
│    • 原始描述 (baseline)   ←→   对抗后描述 (hallucinated)    │
│    • 评估幻觉诱导成功率                                       │
│    • 分析注意力偏移程度                                       │
└─────────────────────────────────────────────────────────────┘
```

**关键脚本说明**:
- **`attack.py`**: 生成对抗样本，通过迭代优化操纵模型注意力机制
- **`generate.py`**: 对原始图片和对抗图片分别生成描述文本
- **对比评估**: 通过比较两组描述，验证幻觉攻击的有效性

> [!NOTE]
> **复现范围**：由于 RTX 4060 (8GB) 显存限制，本指南仅复现 **MiniGPT-4 (Vicuna-7B)** 部分。

---

## 🚀 快速开始 (Quick Start)

### 总体步骤

| 阶段 | 内容 |
|------|------|
| **阶段 0** | [下载项目模型](#阶段-0-下载项目模型-download-project-model) |
| **阶段 1** | [环境配置与依赖安装](#阶段-1-环境配置与依赖安装) |
| **阶段 2** | [下载模型权重](#阶段-2-下载模型权重-download-model-weights) |
| **阶段 3** | [修改配置文件](#阶段-3-修改配置文件-configuration) |
| **阶段 4** | [准备数据集](#阶段-4-准备数据集-data-preparation) |
| **阶段 5** | [运行验证](#阶段-5-运行验证-running--verification) |

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


### 🛠️ 方法 A: 自动化安装 (推荐)
本项目提供了一个专门针对 Windows 11 和 RTX 4060 (8GB) 优化的自动配置脚本，可实现一键安装。请查看[setup_windows.bat](setup_windows.bat)。

#### 脚本功能说明：
*   **环境清理**：自动检查并删除旧的 `mllm` 环境，避免依赖冲突。
*   **Conda 环境**：创建 Python 3.9.20 虚拟环境。
*   **核心库安装**：自动安装 PyTorch 2.0.1 + CUDA 11.8。
*   **依赖安装**：自动安装 `requirements_windows.txt` 及 `CLIP`、`baukit` 等必要组件。
*   **本地库安装**：自动安装修改版的本地 `transformers` 库。

#### 使用说明：
1.  以管理员身份打开 **PowerShell**。
2.  进入项目根目录：`cd "D:\AI PROJEAT\mirage"`
3.  运行命令：`.\setup_windows.bat`
4.  等待脚本完成，期间请确保网络畅通（需访问 GitHub）。

---

### ⌨️ 方法 B: 手动安装 (备选)
如果你希望更精细地控制安装过程，或自动化脚本报错，请按照以下步骤操作。

#### 1.1 创建 Conda 环境
```powershell
# 创建 Python 3.9 环境
conda create -n mllm python=3.9.20
conda activate mllm
```

#### 1.2 安装依赖
```powershell
# 安装基础依赖
pip install -r requirements.txt
```

> [!CAUTION]
> **关于 `requirements.txt` 的重要避坑说明**：
> 原始项目的 `requirements.txt` 包含了大量针对 Linux 服务器环境的配置，在 Windows/Win11 下会报错，原因包括：
> - ✗ 包含 29 个 Linux 硬编码路径链接 (`file:///home/...`)。
> - ✗ 包含 Linux 专用库（如 `uvloop`, `ptyprocess` 等）。
> - ✗ 包含需要 SSH 密钥配置的 Git 依赖。
>
> 因此，**如果你是 Windows 用户，请务必使用下方的 Windows 专用命令**，以避免安装失败。

> [!IMPORTANT]
> 由于原项目是在 Linux/服务器环境上运行，我们为 Win11 优化了依赖文件：[requirements_windows.txt](requirements_windows.txt)
```powershell
# 安装 Windows 专用依赖
pip install -r requirements_windows.txt

# 安装修改版 transformers
python -m pip install -e transformers-4.29.2
```

#### 1.3 补装额外依赖
```powershell
conda activate mllm
pip install sentencepiece accelerate peft timm einops open_clip_torch opencv-python omegaconf webdataset matplotlib pandas
```

#### 1.4 验证安装
使用我们编写的验证脚本[verify_install.py](verify_install.py)检查环境是否配置正确：
```powershell
python verify_install.py
```

> [!TIP]
> 如果出现核心错误，请检查 CUDA 和 PyTorch 版本。
> **重要提醒**：虽然您的显卡驱动可能支持 CUDA 12.x，但本项目依赖 CUDA 11.x 的环境。脚本中已默认安装 CUDA 11.8 版本的 PyTorch 以确保最佳兼容性。推荐环境：CUDA 11.7+ / PyTorch 2.0+。

---

## 阶段 2: 下载模型权重 (Download Model Weights)

### 2.1 Vicuna-7B-v1.5 (~13GB)

#### 方式 1: 使用 HuggingFace CLI (推荐)
```powershell
# 创建目录
mkdir -p "D:\AI PROJEAT\mirage\weights\vicuna"

# 登录 Hugging Face (首次使用需要)
# 获取 Token 链接: https://huggingface.co/settings/tokens
huggingface-cli login

# 下载模型
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir "D:\AI PROJEAT\mirage\weights\vicuna\vicuna-7b-v1.5"
```

#### 方式 2: 手动下载
1. 访问：[HuggingFace - Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
2. 在 "Files and versions" 中下载所有文件。
3. 保存至：`D:\AI PROJEAT\mirage\weights\vicuna\vicuna-7b-v1.5\`

### 2.2 MiniGPT-4 预训练权重 (~36MB)
从网站上（https://github.com/Vision-CAIR/MiniGPT-4?tab=readme-ov-file）找到，下载pretrained_minigpt4_7b.pth
或者直接访问：https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view

> [!WARNING]
> **显存限制提醒**
> - RTX 4060 (8GB) 仅支持 Vicuna-7B 模型
> - 不要下载 Vicuna-13B 或更大的模型
> - 确保使用 `vicuna-7b-v1.5` 对应的 7B 版本以节省显存

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
llama_model: "D:/AI PROJEAT/mirage/weights/vicuna/vicuna-7b-v1.5"
```

> [!TIP]
> 使用绝对路径可以避免路径错误。Windows 用户可以使用正斜杠 `/` 或双反斜杠 `\\`。

---

## 阶段 4: 准备数据集 (Data Preparation)

### 4.1 Hallubench 数据集

#### 下载方式

访问项目网站 [https://huggingface.co/RachelHGF/Mirage-in-the-Eyes](https://huggingface.co/RachelHGF/Mirage-in-the-Eyes) 提供了 Hallubench 数据集的下载方式：
访问这个地址：https://github.com/opendatalab/HA-DPO

目录结构
```powershell
data
├── hadpo/minigpt4/
│  └── desc_data.json
│  └── pope_data.json
```

### 4.2 Visual Genome 数据集（可选）

如需训练或完整复现，请下载 Visual Genome 数据集：

**下载链接：**
- 图像数据集（Part 1）: [VG_100K](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) (~5GB)
- 图像数据集（Part 2）: [VG_100K_2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) (~9GB)
- 元数据: [image_data.json](http://visualgenome.org/static/data/dataset/image_data.json.zip) (~1.69MB)
- 区域描述: [region_descriptions.json](http://visualgenome.org/static/data/dataset/region_descriptions.json.zip) (~121MB)

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

### 5.1 对抗攻击原理 (Attack Mechanism)

#### 核心思想

**Mirage-in-the-Eyes** 利用 **PGD (Projected Gradient Descent)** 迭代优化算法，在图片上添加人眼难以察觉的扰动，通过操纵模型的**注意力机制**诱导产生幻觉输出。

#### 攻击流程

```
原始图片 → [初始化随机扰动] → 迭代优化循环 (30轮)
                                    ↓
         ┌─────────────────────────────────────┐
         │  1. 前向传播 (模型推理)              │
         │  2. 提取注意力矩阵和隐藏状态         │
         │  3. 计算损失 (注意力偏移程度)        │
         │  4. 反向传播求梯度                   │
         │  5. 更新扰动: δ ← δ + α·sign(∇)     │
         └─────────────────────────────────────┘
                                    ↓
                          保存对抗图片
```

#### 关键组件

| 组件 | 作用 | 是否依赖于MLLM |
|------|------|---------|
| **前向传播** | 获取模型对当前扰动图片的响应 | ✅ **必需** |
| **注意力提取** | 分析模型注意力分布 (`attentions`) | ✅ **必需** |
| **损失计算** | 评估注意力偏移程度 | ✅ **必需** |
| **反向传播** | 计算损失对图片像素的梯度 | ✅ **必需** |
| **扰动更新** | 沿梯度方向修改图片 | ❌ 无需 |

> [!IMPORTANT]
> 对抗攻击是一个**优化问题**，每轮迭代都需要完整的模型前向+反向传播，**无法绕过本地模型**。

---

### 5.2 硬件限制与困境 (Hardware Limitations)

#### 显存需求分析

对抗攻击 (`attack.py`) 的显存占用远超普通推理：

| 资源类型 | 占用量 | 说明 |
|---------|--------|------|
| 模型加载 (VIT + Q-Former + Vicuna-7B) | ~13-14 GB | 基础占用 |
| 前向传播激活值 | ~2-3 GB | 中间张量 |
| 反向传播梯度 | ~2-3 GB | 梯度存储 |
| **总计** | **~18-20 GB** | RTX 4060 (8GB) 无法满足 |

#### 遇到的问题

# 显存爆了！！！！！！！！！

在 **RTX 4060 (8GB)** 环境下运行 `attack.py` 会遇到以下错误：

```
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 32.00 MiB (GPU 0; 8.00 GiB total capacity; 
14.39 GiB already allocated; 0 bytes free)
```

---

### 5.3 解决方案：device_map 内存优化 (Solution: device_map Memory Optimization)

#### 问题诊断

在 RTX 4060 (8GB) 环境下，即使是推理 (+generate.py+) 也可能遭遇 OOM：

```
Loading LLAMA...
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**根本原因**：Vicuna-7B 模型本身需要约 13-14 GB 显存，远超 8GB 容量。

#### 解决方案：HuggingFace Accelerate 的 +device_map+

通过启用 **device_map="auto"** 和 **offload_folder**，可以将部分模型层自动分配到 CPU/磁盘：

```python
# minigpt4/models/mini_gpt4.py (已修改)
self.llama_model = LlamaForCausalLM.from_pretrained(
    llama_model,
    torch_dtype=torch.float16,
    device_map="auto",          # 自动分配模型各层到 GPU/CPU
    offload_folder="./offload_cache",  # 临时文件夹用于磁盘 offload
)
```

**工作原理**：
1.  **GPU 层**：将模型的前几层（高频访问）放在 GPU
2.  **CPU 层**：将中间层 offload 到 CPU 内存
3.  **磁盘层**：将末尾层 offload 到磁盘（最慢但节省内存）

#### 配置文件修改

在 +eval_configs/minigpt4_eval.yaml+ 中添加：

```yaml
model:
  arch: mini_gpt4
  # ... 其他配置 ...
  
  # 显存优化：启用模型切分和 CPU offload
  device_map: "auto"
  offload_folder: "./offload_cache"
```

#### 代码修改

为了支持 device_map，需要在 +generate.py+ 和 +test_small_sample.py+ 中添加智能设备分配逻辑：

```python
# 检查 LLM 是否使用了 device_map
llm_model = getattr(model, 'llama_model', None) or getattr(model, 'llm_model', None)

if llm_model is not None and getattr(llm_model, "hf_device_map", None) is not None:
    # LLM 使用了 device_map（可能 offload 到 CPU/磁盘）
    # 只将视觉编码器和 Q-Former 搬到 GPU
    if hasattr(model, 'visual_encoder'):
        model.visual_encoder = model.visual_encoder.to(device)
    if hasattr(model, 'ln_vision'):
        model.ln_vision = model.ln_vision.to(device)
    if hasattr(model, 'Qformer'):
        model.Qformer = model.Qformer.to(device)
    if hasattr(model, 'llama_proj'):
        model.llama_proj = model.llama_proj.to(device)
    if hasattr(model, 'query_tokens'):
        model.query_tokens.data = model.query_tokens.data.to(device)
else:
    # LLM 没用 device_map，整体搬到 GPU
   model = model.to(device)
```

#### 性能与权衡

| 方案 | 推理速度 | 显存占用 | 适用场景 |
|------|---------|---------|---------|
| **全 GPU** |  最快 |  ~14 GB | 高端 GPU (A100/V100) |
| **device_map (GPU+CPU)** |  较快 |  ~4-6 GB | 中端 GPU (RTX 3090/4090) |
| **device_map (GPU+CPU+Disk)** |  较慢 |  ~2-4 GB | **低端 GPU (RTX 4060 8GB)**  |

**实测结果**（RTX 4060 8GB）：
-  **+generate.py+** - 推理成功（每张图片约 20-30 秒）
-  **+test_small_sample.py+** - 验证通过
-  **+attack.py+** - 仍然 OOM（见下文分析）

---

### 5.4 攻击程序失败的根本原因 (Root Cause of Attack Failure)

#### 症状：对抗图片目录为空

运行 +attack.py+ 后检查输出目录：

```powershell
PS> ls outputs/minigpt4_*/adv_images/
# Empty - 没有生成任何 .png 文件！
```

+attack.log+ 也是空的，说明攻击过程中途失败但错误被静默忽略。

#### 直接测试 attack.py

```powershell
python attack.py --model minigpt4 --gpu-id 0 \
  --data-path test_attack_data.json \
  --images-path "data\VG\VG_100K" \
  --save-path "test_adv_output" \
  --generation-mode greedy --eps 2
```

**结果**：

```
Initializing Model
Loading VIT Done
Loading Q-Former Done
Loading LLAMA Done
...
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 256.00 MiB (GPU 0; 8.00 GiB total capacity; 
7.89 GiB already allocated; 0 bytes free; 7.93 GiB reserved)
```

#### 为什么 device_map 对攻击无效？

| 阶段 | 内存需求 | device_map 效果 |
|------|---------|----------------|
| **推理 (generate.py)** | 模型权重 (13-14 GB) + 前向激活值 (1-2 GB) |  **有效** - 可将模型层 offload |
| **攻击 (attack.py)** | 模型权重 + 前向激活值 + **反向传播梯度** (2-3 GB) + **注意力图** (1-2 GB) + **优化器状态** (1 GB) |  **无效** - 梯度和注意力图必须在 GPU |

**关键差异**：

1. **梯度存储**：PGD 攻击需要保存每层的梯度用于反向传播
   ```python
   loss.backward()  # 梯度会占用大量 GPU 显存
   ```

2. **注意力图提取**：需要完整的注意力矩阵用于损失计算
   ```python
   outputs = model.generate(..., output_attentions=True)
   attn_map = outputs.attentions  # 存储所有层的注意力（巨大）
   ```

3. **30 轮迭代**：每轮都需要完整的前向+反向，累积内存压力

**即使使用 device_map，这些中间张量仍然必须在 GPU 上，导致 OOM。**

#### 为什么之前没有发现这个问题？

**静默失败机制**：

1. +run_attack_pipeline.py+ 调用 +attack.py+ 作为子进程
2. 子进程崩溃 (+exit code: 1+)，但父进程没有中断
3. 攻击阶段"完成"（假成功），继续执行生成和评估阶段
4. 用户看到" 流程完成！"但实际上对抗图片根本没生成

**检查方法**：

```powershell
# 检查对抗图片是否真的生成了
Get-ChildItem outputs\*\adv_images\*.png
# 如果为空  攻击失败
```

#### 可能的解决方案（尚未实现）

**选项 1：减少优化复杂度**
- 减少迭代次数（30  10）
- 简化损失函数（去除注意力图提取）
- 使用梯度累积（gradient accumulation）

**选项 2：模型量化**
- 使用 4-bit 量化 (bitsandbytes)
- 牺牲精度换取显存

**选项 3：云端 GPU**
- 使用 Google Colab Pro (A100 40GB)
- AWS/Azure GPU 实例

**选项 4：分布式攻击**
- 将 30 轮迭代分成多个批次
- 每批次清空 GPU cache

> [!CAUTION]
> **当前状态**：在 RTX 4060 (8GB) 上，+attack.py+ **无法成功生成对抗图片**。
> 建议使用至少 16GB 显存的 GPU，或跳过攻击阶段直接验证推理功能。

#### 验证推理功能（跳过攻击）

如果只想验证模型推理和 device_map 功能，可以：

```powershell
# 方法 1: 使用测试脚本（推荐）
python test_small_sample.py --model minigpt4 --num-samples 5

# 方法 2: 跳过攻击阶段
python run_attack_pipeline.py --model minigpt4 --num-samples 5 --skip-attack
```

这将验证：
-  模型加载和 device_map 配置
-  图片预处理和推理流程
-  描述生成功能


