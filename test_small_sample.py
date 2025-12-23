"""
小样本测试脚本 - HallUBench 数据集验证
用于验证数据集完整性和模型推理流程
"""

import argparse
import os
import json
import random
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from torchvision import transforms
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ========================================
#             配置参数
# ========================================
MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
}

# 数据路径配置
DATA_PATH = "data/hadpo/minigpt4/desc_data.json"
VG_IMAGE_PATHS = [
    "data/VG/VG_100K",
    "data/VG/VG_100K_2",
]


def setup_seeds(config, seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def find_image_path(image_id, vg_paths):
    """
    根据 image_id 查找图片路径
    图片可能在 VG_100K 或 VG_100K_2 文件夹中
    """
    for vg_path in vg_paths:
        # 尝试 .jpg 扩展名
        jpg_path = os.path.join(vg_path, f"{image_id}.jpg")
        if os.path.exists(jpg_path):
            return jpg_path
        # 尝试 .png 扩展名
        png_path = os.path.join(vg_path, f"{image_id}.png")
        if os.path.exists(png_path):
            return png_path
    return None


def load_sample_data(data_path, num_samples=5):
    """加载小样本数据"""
    print(f"[INFO] 正在加载数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机采样
    if len(data) > num_samples:
        samples = random.sample(data, num_samples)
    else:
        samples = data
    
    print(f"[INFO] 已加载 {len(samples)} 个样本 (总共 {len(data)} 个)")
    return samples


def validate_images(samples, vg_paths):
    """验证图片是否存在"""
    print("\n[INFO] 正在验证图片路径...")
    valid_samples = []
    missing_images = []
    
    for sample in samples:
        image_id = sample["image_id"]
        image_path = find_image_path(image_id, vg_paths)
        
        if image_path:
            sample["image_path"] = image_path
            valid_samples.append(sample)
            print(f"  ✓ 图片 {image_id} 找到: {image_path}")
        else:
            missing_images.append(image_id)
            print(f"  ✗ 图片 {image_id} 未找到")
    
    print(f"\n[INFO] 验证完成: {len(valid_samples)}/{len(samples)} 张图片可用")
    if missing_images:
        print(f"[WARNING] 缺失图片: {missing_images}")
    
    return valid_samples


def initialize_model(model_name, gpu_id=0):
    """初始化模型"""
    print(f"\n[INFO] 正在初始化模型: {model_name}")
    
    # 创建参数对象
    class Args:
        def __init__(self):
            self.cfg_path = MODEL_EVAL_CONFIG_PATH[model_name]
            self.gpu_id = gpu_id
            self.options = None
    
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    cfg = Config(args)
    setup_seeds(cfg)
    
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}")
    
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    
    # 获取预处理器
    processor_cfg = cfg.get_config().preprocess
    processor_cfg.vis_processor.eval.do_normalize = False
    vis_processors, txt_processors = load_preprocess(processor_cfg)
    
    print("[INFO] 模型初始化完成!")
    return model, vis_processors, device


def run_inference(model, vis_processors, device, samples, model_name, num_beams=3):
    """运行推理"""
    print(f"\n[INFO] 开始推理测试 ({len(samples)} 个样本)...")
    
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)
    
    results = []
    template = INSTRUCTION_TEMPLATE[model_name]
    question = "Please describe this image in detail."
    prompt = template.replace("<question>", question)
    
    for idx, sample in enumerate(samples):
        image_id = sample["image_id"]
        image_path = sample["image_path"]
        
        print(f"\n[{idx+1}/{len(samples)}] 处理图片: {image_id}")
        
        # 加载并预处理图片
        image = Image.open(image_path).convert("RGB")
        image_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
        
        # 生成描述
        with torch.inference_mode():
            with torch.no_grad():
                output = model.generate(
                    {"image": norm(image_tensor), "prompt": prompt},
                    use_nucleus_sampling=False,
                    num_beams=num_beams,
                    max_new_tokens=512,
                    output_attentions=False,
                    opera_decoding=False,
                )
        
        generated_text = output[0] if output else ""
        
        result = {
            "image_id": image_id,
            "image_path": image_path,
            "generated": generated_text,
            "chosen": sample.get("chosen", []),
            "rejected": sample.get("rejected", []),
        }
        results.append(result)
        
        # 打印对比结果
        print(f"  生成描述: {generated_text[:200]}...")
        print(f"  Chosen样例: {sample['chosen'][0][:100]}..." if sample.get('chosen') else "  无Chosen样例")
    
    return results


def save_results(results, output_path="test_results.json"):
    """保存测试结果"""
    print(f"\n[INFO] 保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[INFO] 结果已保存!")


def main():
    parser = argparse.ArgumentParser(description="小样本测试脚本")
    parser.add_argument("--model", type=str, default="minigpt4", 
                        choices=["minigpt4", "instructblip"],
                        help="选择模型类型")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--num-samples", type=int, default=5, help="测试样本数量")
    parser.add_argument("--output", type=str, default="test_results.json", 
                        help="输出结果文件路径")
    parser.add_argument("--validate-only", action="store_true", 
                        help="仅验证数据,不运行模型推理")
    parser.add_argument("--num-beams", type=int, default=3, help="Beam search 数量")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("小样本测试脚本 - HallUBench 数据集验证")
    print("=" * 60)
    
    # Step 1: 加载样本数据
    samples = load_sample_data(DATA_PATH, args.num_samples)
    
    # Step 2: 验证图片路径
    valid_samples = validate_images(samples, VG_IMAGE_PATHS)
    
    if not valid_samples:
        print("[ERROR] 没有可用的样本,请检查数据路径!")
        return
    
    if args.validate_only:
        print("\n[INFO] 仅验证模式,跳过模型推理")
        return
    
    # Step 3: 初始化模型
    model, vis_processors, device = initialize_model(args.model, args.gpu_id)
    
    # Step 4: 运行推理
    results = run_inference(
        model, vis_processors, device, 
        valid_samples, args.model, args.num_beams
    )
    
    # Step 5: 保存结果
    save_results(results, args.output)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
