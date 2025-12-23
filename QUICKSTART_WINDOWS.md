# Mirage-in-the-Eyes 快速开始指南

**硬件配置**: RTX 4060 Laptop (8GB VRAM)  
**复现方案**: A - 最小可行复现（仅 MiniGPT-4）

---

## 📋 文件清单

本次已为您创建以下 Windows 兼容文件：

| 文件 | 说明 |
|------|------|
| `requirements_windows.txt` | Windows 兼容的依赖列表（已清理 29 个 Linux 路径依赖） |
| `setup_windows.bat` | 一键安装脚本（自动化环境配置） |
| `verify_install.py` | 环境验证脚本（检查所有依赖） |

---

## 🚀 快速开始（3 种方式）

### 方式 1: 一键安装（推荐）

```powershell
# 在项目根目录运行
.\setup_windows.bat
```

**完成后验证**:
```powershell
conda activate mllm
python verify_install.py
```

---

### 方式 2: 手动安装

#### 步骤 1: 创建环境
```powershell
conda create -n mllm python=3.9.20
conda activate mllm
```

#### 步骤 2: 安装 PyTorch (CUDA 11.8)
```powershell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ **重要**: 虽然您的驱动支持 CUDA 12.7，但项目使用 CUDA 11.x 库，需安装 CUDA 11.8 版本的 PyTorch

#### 步骤 3: 安装依赖
```powershell
pip install -r requirements_windows.txt
```

#### 步骤 4: 安装 Git 依赖
```powershell
# CLIP
pip install git+https://github.com/openai/CLIP.git

# baukit
pip install git+https://github.com/davidbau/baukit.git
```

#### 步骤 5: 安装本地 transformers
```powershell
# 解压
powershell -command "Expand-Archive -Force transformers-4.29.2.zip ."

# 安装
cd transformers-4.29.2
python -m pip install -e .
cd ..
```

#### 步骤 6: 验证
```powershell
python verify_install.py
```

---

### 方式 3: 使用原始 requirements.txt（不推荐）

**问题**: 原始文件包含以下不兼容内容：
- ✗ 29 个 Linux 路径依赖 (`file:///home/...`)
- ✗ Linux 专用库（`uvloop`, `ptyprocess`）
- ✗ SSH Git 依赖（需配置 SSH 密钥）

**如需使用，必须手动修改后安装**

---

## 📦 下一步：下载模型

### Vicuna-7B-v0 (~13GB)

**方式 1: 使用 Hugging Face CLI**
```powershell
pip install huggingface_hub

# 创建模型目录
mkdir -p "D:\AI PROJEAT\mirage\weights\vicuna"

# 下载（需要 Hugging Face 账号）
huggingface-cli login
huggingface-cli download lmsys/vicuna-7b-v0 --local-dir "D:\AI PROJEAT\mirage\weights\vicuna\vicuna-7b-v0"
```

**方式 2: 手动下载**
1. 访问: https://huggingface.co/lmsys/vicuna-7b-v0
2. 下载所有文件到 `D:\AI PROJEAT\mirage\weights\vicuna\vicuna-7b-v0\`

---

### MiniGPT-4 预训练权重 (~5GB)

1. 访问官方仓库: https://github.com/Vision-CAIR/MiniGPT-4
2. 在 README 中找到权重下载链接
3. 下载 `pretrained_minigpt4_7b.pth`
4. 保存到: `D:\AI PROJEAT\mirage\weights\minigpt4\pretrained_minigpt4_7b.pth`

---

## ⚙️ 配置模型路径

### 修改 `eval_configs/minigpt4_eval.yaml`

```yaml
# 第 8 行，修改为:
ckpt: 'D:/AI PROJEAT/mirage/weights/minigpt4/pretrained_minigpt4_7b.pth'
```

### 修改 `minigpt4/configs/models/minigpt4_vicuna0.yaml`

```yaml
# 第 18 行，修改为:
llama_model: "D:/AI PROJEAT/mirage/weights/vicuna/vicuna-7b-v0"
```

---

## 🧪 测试运行

### 1. 准备测试数据

创建 `test_data.json`:
```json
[
  {
    "image_id": "test_001",
    "question": "Please describe this image in detail."
  }
]
```

创建测试图像目录并放入一张图片 `test_001.png`:
```powershell
mkdir test_images
# 复制一张测试图片到 test_images\test_001.png
```

### 2. 运行推理测试

```powershell
conda activate mllm

python generate.py `
  --model minigpt4 `
  --gpu-id 0 `
  --data-path test_data.json `
  --images-path test_images `
  --response-path test_response.json `
  --generation-mode greedy
```

### 3. 查看结果

```powershell
type test_response.json
```

---

## ⚠️ 注意事项（8GB 显存配置）

1. **仅使用 Vicuna-7B**: 不要尝试 13B 模型，显存不足
2. **关闭其他程序**: 运行前关闭占用显存的程序（浏览器、游戏等）
3. **使用 batch_size=1**: 默认已是 1，不要修改
4. **监控显存**: 运行时可通过 `nvidia-smi` 监控显存使用

---

## 🐛 常见问题

### Q1: PyTorch CUDA 不可用
```powershell
# 检查
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，重新安装 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### Q2: CUDA out of memory
- 关闭所有其他占用显存的程序
- 在 Windows 任务管理器中查看 GPU 使用情况
- 考虑降低图像分辨率（修改配置文件中的 `image_size`）

### Q3: transformers 导入失败
```powershell
cd transformers-4.29.2
pip install -e . --force-reinstall --no-deps
cd ..
```

### Q4: CLIP 安装失败
需要 Git 支持，确保已安装 Git:
```powershell
# 检查 Git
git --version

# 如未安装，访问 https://git-scm.com/download/win
```

---

## 📊 预期时间

| 步骤 | 时间 |
|------|------|
| 环境安装 | 30-60 分钟 |
| 模型下载 | 1-2 小时（取决于网速） |
| 配置修改 | 10 分钟 |
| 测试运行 | 5-10 分钟 |
| **总计** | **约 2-3.5 小时** |

---

## 📞 获取帮助

如遇到问题：
1. 查看 `verify_install.py` 输出的错误信息
2. 检查 `implementation_plan.md` 中的详细说明
3. 确认 GPU 驱动和 CUDA 版本

---

**祝您复现顺利！** 🎉
