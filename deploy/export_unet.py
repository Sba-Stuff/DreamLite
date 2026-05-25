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

import os
import sys
import torch
import numpy as np
import coremltools as ct
from pathlib import Path

# ========= Config ============ 
MODEL_PATH = "models/DreamLite-mobile"
OUTPUT_DIR = Path("./exported_models")
OUTPUT_NAME = "unet.mlpackage"
# =============================

OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load UNet
print("Loading UNet...")
from dreamlite import DreamLiteMobilePipeline

unet = DreamLiteMobilePipeline.from_pretrained(MODEL_PATH, sub_folder='unet', dtype=torch.float32)
unet.eval()
for p in unet.parameters():
    p.requires_grad = False
print(f"  UNet loaded: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M params")

# 2. 创建 Wrapper —— 把 dict 参数拆成独立的 tensor 参数
class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, encoder_attention_mask, time_ids):
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]

wrapper = UNetWrapper(unet)
wrapper.eval()

# 3. 准备示例输入
#    sample:                 [1, 4, 128, 256]  (in-context, width 拼接)
#    timestep:               [1]
#    encoder_hidden_states:  [1, 77, 2048]     (Qwen3-VL 输出, 用 77 作为默认)
#    encoder_attention_mask:  [1, 77]
#    time_ids:               [1, 2]

SEQ_DEFAULT = 77
dummy_inputs = (
    torch.randn(1, 4, 128, 256),           # sample
    torch.tensor([500.0]),                   # timestep
    torch.randn(1, SEQ_DEFAULT, 2048),       # encoder_hidden_states
    torch.ones(1, SEQ_DEFAULT),              # encoder_attention_mask
    torch.tensor([[1024.0, 1024.0]]),        # time_ids [width, height]
)

print(f"  Input shapes:")
print(f"    sample:                {dummy_inputs[0].shape}")
print(f"    timestep:              {dummy_inputs[1].shape}")
print(f"    encoder_hidden_states: {dummy_inputs[2].shape}")
print(f"    encoder_attention_mask:{dummy_inputs[3].shape}")
print(f"    time_ids:              {dummy_inputs[4].shape}")

# 4. Test PyTorch inference
print("\nTesting PyTorch inference...")
with torch.no_grad():
    test_output = wrapper(*dummy_inputs)
print(f"  Output shape: {test_output.shape}")  # 应该是 [1, 4, 128, 256]

# 5. TorchScript Trace
print("\nTracing model (this may take a minute)...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy_inputs, strict=False)
traced = torch.jit.freeze(traced)
print("  Trace successful!")

# 6. Convert to CoreML
print("\nConverting to CoreML (this may take several minutes)...")

SEQ_MIN = 10
SEQ_MAX = 512
seq_range = ct.RangeDim(lower_bound=SEQ_MIN, upper_bound=SEQ_MAX, default=SEQ_DEFAULT)

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="sample",                  shape=(1, 4, 128, 256)),
        ct.TensorType(name="timestep",                 shape=(1,)),
        ct.TensorType(name="encoder_hidden_states",    shape=(1, seq_range, 2048)),
        ct.TensorType(name="encoder_attention_mask",   shape=(1, seq_range)),
        ct.TensorType(name="time_ids",                 shape=(1, 2)),
    ],
    outputs=[ct.TensorType(name="noise_pred")],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.ALL,
)

# 7. 保存
save_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
mlmodel.save(str(save_path))
size_mb = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file()) / 1024**2

print(f"\n✅ UNet exported successfully!")
print(f"   Path: {save_path}")
print(f"   Size: {size_mb:.1f} MB")