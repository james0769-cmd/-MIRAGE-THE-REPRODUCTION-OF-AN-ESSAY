import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm
import minigpt4.models.blip2_vicuna_instruct as m


from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="instructblip", help="instructblip/minigpt4/llava-1.5/shikra")
parser.add_argument("--gpu-id", type=int, help="specify the gpu to load the model.")
parser.add_argument("--data-path", type=str, help="path to dataset")
parser.add_argument("--images-path", type=str, help="path to vanilla images or adv images")
parser.add_argument("--response-path", type=str, help='path to responses')
parser.add_argument("--mode", type=str, default="vanilla")
parser.add_argument("--generation-mode", type=str, default="beam", help="beam/greedy/nucleus")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")
parser.add_argument("--beam", type=int, default=3)
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=3)
parser.add_argument("--penalty_weights", type=float, default=1.0)
args = parser.parse_known_args()[0]


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

model_config = cfg.model_cfg
# --- PATCH: make --options also update model_cfg (needed for offload_folder) ---
if args.options is not None:
    for opt in args.options:
        if "=" not in opt:
            continue
        k, v = opt.split("=", 1)
        # we expect keys like "model.llm_model" / "model.offload_folder"
        if k.startswith("model."):
            k = k[len("model."):]
        model_config[k] = v
# --- end patch ---
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config)

# 如果 LLM 没用 device_map，就整体搬到 GPU
if getattr(model.llm_model, "hf_device_map", None) is None:
    model = model.to(device)
else:
    # LLM 用了 device_map（可能 offload），只搬 vision/qformer
    model.visual_encoder = model.visual_encoder.to(device)
    model.ln_vision = model.ln_vision.to(device)
    model.Qformer = model.Qformer.to(device)
    model.llm_proj = model.llm_proj.to(device)
    if hasattr(model, "query_tokens"):
        model.query_tokens.data = model.query_tokens.data.to(device)

# --- end patch ---
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
print("Done!")

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

if args.generation_mode == "nucleus":
    use_nucleus_sampling = True
else:
    use_nucleus_sampling = False

if args.generation_mode == "greedy":
    args.beam = 1
    
print("use_nucleus_sampling", use_nucleus_sampling)
print("num_beams", args.beam)

data = json.load(open(args.data_path))
for _data in tqdm.tqdm(data):
    image_id = _data["image_id"]
    image_path = f'{args.images_path}/{image_id}.png'
    if os.path.exists(image_path) == False:
        print(f'{image_path} not exist!')
        continue
    else:
        print(image_path)
    
    image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](image).unsqueeze(0)
    image = image.to(device)
    
    qu = "Please describe this image in detail."
    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt":qu}, 
                use_nucleus_sampling=use_nucleus_sampling, 
                num_beams=args.beam,
                max_new_tokens=512,
                output_attentions=False,
            )
            
    with open(args.response_path, "a") as f:
        json.dump({image_id: out[0]}, f)
        f.write('\n')