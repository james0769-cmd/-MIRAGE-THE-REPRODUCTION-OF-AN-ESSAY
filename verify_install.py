"""
验证 Mirage-in-the-Eyes 环境安装
适用于: Windows 11, RTX 4060 Laptop (8GB)
"""

import sys
import os

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_import(module_name, package_name=None):
    """检查模块是否可以导入"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name:30s} [已安装]")
        return True
    except ImportError as e:
        print(f"✗ {package_name:30s} [缺失] - {str(e)[:40]}")
        return False

def main():
    print_section("Mirage-in-the-Eyes 环境验证")
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    
    all_ok = True
    
    # ========================================
    # 核心库检查
    # ========================================
    print_section("1. 核心深度学习库")
    
    # PyTorch
    torch_ok = check_import("torch", "PyTorch")
    if torch_ok:
        import torch
        print(f"  - PyTorch 版本: {torch.__version__}")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA 版本: {torch.version.cuda}")
            print(f"  - GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU 数量: {torch.cuda.device_count()}")
            
            # 显存检查
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  - 总显存: {total_memory:.2f} GB")
            
            if total_memory < 10:
                print(f"  ⚠️  警告: 显存较少 ({total_memory:.2f}GB)，建议仅使用 Vicuna-7B")
        else:
            print("  ⚠️  警告: CUDA 不可用，无法使用 GPU 加速")
            all_ok = False
    else:
        all_ok = False
    
    check_import("torchvision", "torchvision")
    
    # ========================================
    # Transformers 生态
    # ========================================
    print_section("2. Transformers 生态")
    
    transformers_ok = check_import("transformers", "transformers (本地版本)")
    if transformers_ok:
        import transformers
        print(f"  - transformers 版本: {transformers.__version__}")
        if transformers.__version__ != "4.29.2":
            print(f"  ⚠️  警告: 期望版本 4.29.2，当前为 {transformers.__version__}")
    else:
        all_ok = False
    
    check_import("tokenizers", "tokenizers")
    check_import("sentencepiece", "sentencepiece")
    check_import("accelerate", "accelerate")
    check_import("peft", "peft")
    
    # ========================================
    # 视觉库
    # ========================================
    print_section("3. 计算机视觉库")
    
    clip_ok = check_import("clip", "CLIP")
    if not clip_ok:
        print("  ⚠️  CLIP 未安装，请运行: pip install git+https://github.com/openai/CLIP.git")
        all_ok = False
    
    check_import("timm", "timm")
    check_import("einops", "einops")
    check_import("open_clip", "open_clip_torch")
    check_import("cv2", "opencv-python")
    
    # ========================================
    # BLIP/LAVIS 依赖
    # ========================================
    print_section("4. BLIP/LAVIS 依赖")
    
    check_import("omegaconf", "omegaconf")
    check_import("webdataset", "webdataset")
    
    # ========================================
    # MiniGPT-4 项目模块
    # ========================================
    print_section("5. MiniGPT-4 项目模块")
    
    # 检查是否在正确目录
    if os.path.exists("minigpt4"):
        print("✓ minigpt4 目录存在")
        
        # 尝试导入
        sys.path.insert(0, os.getcwd())
        minigpt4_ok = check_import("minigpt4", "minigpt4 (项目模块)")
        
        if minigpt4_ok:
            try:
                from minigpt4.common.config import Config
                print("✓ minigpt4.common.config         [可导入]")
            except Exception as e:
                print(f"✗ minigpt4.common.config         [导入失败] - {e}")
                all_ok = False
            
            try:
                from minigpt4.models import load_preprocess
                print("✓ minigpt4.models                [可导入]")
            except Exception as e:
                print(f"✗ minigpt4.models                [导入失败] - {e}")
                all_ok = False
    else:
        print("✗ minigpt4 目录不存在")
        print("  请确保在项目根目录 d:\\AI PROJEAT\\mirage 下运行")
        all_ok = False
    
    # ========================================
    # 其他工具库
    # ========================================
    print_section("6. 其他工具库")
    
    check_import("numpy", "numpy")
    check_import("PIL", "Pillow")
    check_import("matplotlib", "matplotlib")
    check_import("pandas", "pandas")
    
    # ========================================
    # 总结
    # ========================================
    print_section("验证总结")
    
    if all_ok:
        print("✓ 所有核心组件已正确安装！")
        print("\n下一步:")
        print("  1. 下载 Vicuna-7B 模型")
        print("  2. 下载 MiniGPT-4 预训练权重")
        print("  3. 修改配置文件中的模型路径")
        print("  4. 准备测试数据")
        print("\n详细步骤请参考: implementation_plan.md")
    else:
        print("✗ 部分组件缺失或配置有误")
        print("\n请检查上述错误并重新安装缺失的库")
        print("如需帮助，请参考 requirements_windows.txt 中的安装说明")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
