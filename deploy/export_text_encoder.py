# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch

# ========= Config, Qwen3-VL-2B, Real Path ============ 
MODEL_PATH = Path("Qwen/Qwen3-VL-2B-Instruct")
OUTPUT_DIR = Path("./exported_models/text_encoder")
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Text Encoder, BF16 → FP16
print("Loading Text Encoder")
state_dict = load_file(str(MODEL_PATH / "model.safetensors"))
print(f"  Loaded {len(state_dict)} tensors")

# check original dtype
sample_key = list(state_dict.keys())[0]
print(f"  Original dtype: {state_dict[sample_key].dtype}")

# convert to FP16
print("Converting to FP16...")
fp16_state_dict = {}
for key, tensor in state_dict.items():
    fp16_state_dict[key] = tensor.to(torch.float16)

# 保存
print("Saving FP16 weights...")
save_file(fp16_state_dict, str(OUTPUT_DIR / "model.safetensors"))
size_gb = (OUTPUT_DIR / "model.safetensors").stat().st_size / 1024**3
print(f"  Saved: {OUTPUT_DIR / 'model.safetensors'} ({size_gb:.2f} GB)")

# 2. copy all necessary config files
files_to_copy = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
]

print("\nCopying config files...")
copied = 0
for fname in files_to_copy:
    src = QWEN_PATH / fname
    if src.exists():
        shutil.copy2(src, OUTPUT_DIR / fname)
        copied += 1
        print(f"  ✓ {fname}")
    else:
        print(f"  ✗ {fname} (not found, skipping)")

print(f"\n✅ Text Encoder preparation complete!")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Weights: {size_gb:.2f} GB (FP16)")
print(f"   Config files: {copied}")
print(f"\n📋 Next step: Transfer this folder to your Mac for MLX 4-bit quantization")

"""
On MacOS, run the following command to quantize the model:
python3 -m mlx_vlm.convert \
    --hf-path ./exported_models/text_encoder \
    --mlx-path ./exported_models/text_encoder_mlx_4bit \
    -q \
    --q-bits 4 \
    --q-group-size 64
"""