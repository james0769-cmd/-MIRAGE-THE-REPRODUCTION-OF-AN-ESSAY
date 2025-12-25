#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mirage-in-the-Eyes: 完整攻击流程自动化脚本
=========================================

本脚本自动执行完整的幻觉攻击流程:
1. 对抗攻击 (attack.py) - 生成对抗图片
2. 描述生成 (generate.py) - 生成原始和对抗图片的描述
3. 效果评估 (eval/json_eval.py) - 评估幻觉诱导效果

使用方法:
    python run_attack_pipeline.py --model minigpt4 --num-samples 5

作者: Mirage 复现项目
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from datetime import datetime
from pathlib import Path


# ==================== 配置区域 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 默认路径配置
DEFAULT_CONFIG = {
    # 数据路径
    "data_path": PROJECT_ROOT / "data" / "hadpo" / "minigpt4" / "desc_data.json",
    
    # 图片路径 (Visual Genome)
    "images_path": PROJECT_ROOT / "data" / "VG" / "VG_100K",
    
    # 输出路径
    "output_dir": PROJECT_ROOT / "outputs",
    
    # 攻击参数
    "eps": 2,  # 扰动强度 (0-255)
    "generation_mode": "greedy",  # beam/greedy/nucleus
    "beam": 1,
    
    # GPU
    "gpu_id": 0,
}


# ==================== 工具函数 ====================

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def timestamp():
    """生成时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(cmd, desc="Running command"):
    """运行命令并实时打印输出"""
    print(f"\n{'='*60}")
    print(f"🚀 {desc}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=PROJECT_ROOT
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ 命令执行失败，返回码: {process.returncode}")
            return False
        
        print(f"\n✅ {desc} 完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 命令执行错误: {e}")
        return False


def create_sample_data(data_path, num_samples, output_path):
    """从完整数据集中提取样本数据"""
    print(f"\n📋 准备样本数据 (共 {num_samples} 个样本)...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是字典格式 (如 test_1sample.json)
    if isinstance(data, dict) and "value" in data:
        data = data["value"]
    
    # 提取样本
    samples = data[:num_samples]
    
    # 保存样本数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存 {len(samples)} 个样本到: {output_path}")
    
    # 返回 image_ids
    image_ids = [s["image_id"] for s in samples]
    return image_ids


def check_images_exist(image_ids, images_path):
    """检查图片文件是否存在"""
    print(f"\n🖼️ 检查图片文件...")
    
    missing = []
    found = []
    
    for img_id in image_ids:
        # 尝试不同的扩展名
        for ext in ['.jpg', '.png', '.jpeg']:
            img_path = Path(images_path) / f"{img_id}{ext}"
            if img_path.exists():
                found.append(img_id)
                break
        else:
            # 也检查 VG_100K_2 目录
            for ext in ['.jpg', '.png', '.jpeg']:
                img_path = Path(images_path).parent / "VG_100K_2" / f"{img_id}{ext}"
                if img_path.exists():
                    found.append(img_id)
                    break
            else:
                missing.append(img_id)
    
    print(f"  找到: {len(found)}/{len(image_ids)} 张图片")
    
    if missing:
        print(f"  ⚠️ 缺失图片: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    return found, missing


def print_summary(args, output_dir, adv_images_dir, vanilla_response, adv_response):
    """打印运行摘要"""
    print("\n")
    print("=" * 60)
    print("📊 攻击流程完成摘要")
    print("=" * 60)
    print(f"""
模型: {args.model}
样本数量: {args.num_samples}
扰动强度 (eps): {args.eps}
生成模式: {args.generation_mode}

输出目录: {output_dir}
├── sample_data.json     # 样本数据
├── adv_images/          # 对抗图片
├── vanilla_response.json # 原始图片描述
└── adv_response.json    # 对抗图片描述

下一步:
1. 查看对抗图片: {adv_images_dir}
2. 对比描述结果:
   - 原始: {vanilla_response}
   - 对抗: {adv_response}
3. 运行评估 (可选):
   python eval/json_eval.py --json-file {adv_response} --bench-path {output_dir}/sample_data.json
""")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="Mirage-in-the-Eyes: 完整攻击流程自动化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 MiniGPT-4 对 5 个样本进行攻击
  python run_attack_pipeline.py --model minigpt4 --num-samples 5
  
  # 使用 InstructBLIP，自定义扰动强度
  python run_attack_pipeline.py --model instructblip --num-samples 10 --eps 4
  
  # 仅生成描述 (跳过攻击，使用已有对抗图片)
  python run_attack_pipeline.py --model minigpt4 --skip-attack
        """
    )
    
    # 模型选择
    parser.add_argument("--model", type=str, default="minigpt4",
                        choices=["minigpt4", "instructblip", "llava-1.5", "shikra"],
                        help="目标模型 (默认: minigpt4)")
    
    # 样本数量
    parser.add_argument("--num-samples", type=int, default=5,
                        help="攻击样本数量 (默认: 5)")
    
    # 攻击参数
    parser.add_argument("--eps", type=float, default=2,
                        help="扰动强度 ε (0-255, 默认: 2)")
    parser.add_argument("--generation-mode", type=str, default="greedy",
                        choices=["beam", "greedy", "nucleus"],
                        help="生成模式 (默认: greedy)")
    parser.add_argument("--beam", type=int, default=1,
                        help="Beam search 宽度 (默认: 1)")
    
    # GPU
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID (默认: 0)")
    
    # 流程控制
    parser.add_argument("--skip-attack", action="store_true",
                        help="跳过攻击步骤 (使用已有对抗图片)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="跳过生成步骤")
    parser.add_argument("--skip-eval", action="store_true",
                        help="跳过评估步骤")
    
    # 路径配置
    parser.add_argument("--data-path", type=str, default=None,
                        help="数据集路径 (默认: data/hadpo/minigpt4/desc_data.json)")
    parser.add_argument("--images-path", type=str, default=None,
                        help="图片路径 (默认: data/VG/VG_100K)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: outputs/<timestamp>)")
    
    args = parser.parse_args()
    
    # ==================== 初始化路径 ====================
    
    # 设置默认路径
    data_path = Path(args.data_path) if args.data_path else DEFAULT_CONFIG["data_path"]
    images_path = Path(args.images_path) if args.images_path else DEFAULT_CONFIG["images_path"]
    
    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_CONFIG["output_dir"] / f"{args.model}_{timestamp()}"
    
    ensure_dir(output_dir)
    
    # 定义各阶段输出路径
    sample_data_path = output_dir / "sample_data.json"
    adv_images_dir = output_dir / "adv_images"
    vanilla_response_path = output_dir / "vanilla_response.json"
    adv_response_path = output_dir / "adv_response.json"
    attack_log_path = output_dir / "attack.log"
    
    ensure_dir(adv_images_dir)
    
    # ==================== 打印配置 ====================
    
    print("\n" + "=" * 60)
    print("🎯 Mirage-in-the-Eyes: 幻觉攻击流程")
    print("=" * 60)
    print(f"""
配置信息:
  模型: {args.model}
  样本数量: {args.num_samples}
  扰动强度 (eps): {args.eps}
  生成模式: {args.generation_mode}
  GPU: {args.gpu_id}
  
路径:
  数据集: {data_path}
  图片目录: {images_path}
  输出目录: {output_dir}
""")
    
    # ==================== 检查文件 ====================
    
    if not data_path.exists():
        print(f"❌ 数据集文件不存在: {data_path}")
        print("请确保已下载 Hallubench 数据集")
        sys.exit(1)
    
    if not images_path.exists():
        print(f"❌ 图片目录不存在: {images_path}")
        print("请确保已下载 Visual Genome 数据集")
        sys.exit(1)
    
    # ==================== 准备样本数据 ====================
    
    image_ids = create_sample_data(data_path, args.num_samples, sample_data_path)
    found_ids, missing_ids = check_images_exist(image_ids, images_path)
    
    if not found_ids:
        print("❌ 没有找到任何图片文件!")
        print(f"请确保 Visual Genome 图片已放置在: {images_path}")
        sys.exit(1)
    
    # 如果有缺失图片，更新样本数据
    if missing_ids:
        print(f"⚠️ 将使用 {len(found_ids)} 张可用图片继续...")
        # 过滤掉缺失图片的样本
        with open(sample_data_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        samples = [s for s in samples if s["image_id"] in found_ids]
        with open(sample_data_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    
    # ==================== 阶段 1: 对抗攻击 ====================
    
    if not args.skip_attack:
        print("\n" + "=" * 60)
        print("📌 阶段 1: 对抗攻击")
        print("=" * 60)
        
        attack_cmd = [
            sys.executable, "attack.py",
            "--model", args.model,
            "--gpu-id", str(args.gpu_id),
            "--data-path", str(sample_data_path),
            "--images-path", str(images_path),
            "--save-path", str(adv_images_dir),
            "--log-path", str(attack_log_path),
            "--generation-mode", args.generation_mode,
            "--eps", str(args.eps),
            "--beam", str(args.beam),
        ]
        
        success = run_command(attack_cmd, "生成对抗图片")
        if not success:
            print("⚠️ 攻击阶段可能遇到问题，请检查日志")
    else:
        print("\n⏭️ 跳过攻击阶段")
    
    # ==================== 阶段 2: 描述生成 ====================
    
    if not args.skip_generate:
        print("\n" + "=" * 60)
        print("📌 阶段 2: 描述生成")
        print("=" * 60)
        
        # 2.1 生成原始图片描述 (Baseline)
        print("\n📝 生成原始图片描述...")
        vanilla_cmd = [
            sys.executable, "generate.py",
            "--model", args.model,
            "--gpu-id", str(args.gpu_id),
            "--data-path", str(sample_data_path),
            "--images-path", str(images_path),
            "--response-path", str(vanilla_response_path),
            "--mode", "vanilla",
            "--generation-mode", args.generation_mode,
            "--beam", str(args.beam),
        ]
        
        run_command(vanilla_cmd, "生成原始图片描述")
        
        # 2.2 生成对抗图片描述 (Hallucinated)
        print("\n📝 生成对抗图片描述...")
        adv_cmd = [
            sys.executable, "generate.py",
            "--model", args.model,
            "--gpu-id", str(args.gpu_id),
            "--data-path", str(sample_data_path),
            "--images-path", str(adv_images_dir),
            "--response-path", str(adv_response_path),
            "--mode", "adv",
            "--generation-mode", args.generation_mode,
            "--beam", str(args.beam),
        ]
        
        run_command(adv_cmd, "生成对抗图片描述")
    else:
        print("\n⏭️ 跳过生成阶段")
    
    # ==================== 阶段 3: 效果评估 ====================
    
    if not args.skip_eval:
        print("\n" + "=" * 60)
        print("📌 阶段 3: 效果评估")
        print("=" * 60)
        
        eval_log_path = output_dir / "eval.log"
        
        eval_cmd = [
            sys.executable, str(PROJECT_ROOT / "eval" / "json_eval.py"),
            "--json-file", str(adv_response_path),
            "--bench-path", str(sample_data_path),
            "--log-path", str(eval_log_path),
        ]
        
        # 检查评估脚本是否存在
        if (PROJECT_ROOT / "eval" / "json_eval.py").exists():
            run_command(eval_cmd, "评估幻觉效果")
        else:
            print("⚠️ 评估脚本不存在，跳过评估阶段")
    else:
        print("\n⏭️ 跳过评估阶段")
    
    # ==================== 完成 ====================
    
    print_summary(args, output_dir, adv_images_dir, vanilla_response_path, adv_response_path)
    
    print("\n🎉 流程完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
