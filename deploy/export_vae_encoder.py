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
OUTPUT_NAME = "vae_encoder.mlpackage"
# =============================

OUTPUT_DIR.mkdir(exist_ok=True)

# 1. 加载 VAE
print("Loading VAE...")
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

vae = AutoencoderTiny.from_pretrained(MODEL_PATH, sub_folder='vae', torch_dtype=torch.float32)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False
print(f"  VAE loaded: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M params")

# 2. Create Wrapper
class VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        latents = self.vae.encode(image, return_dict=False)[0]
        return latents

wrapper = VAEEncoderWrapper(vae)
wrapper.eval()

# 3. Dummy Input Shape: [1, 3, 1024, 1024], Range [-1, 1]
dummy_image = torch.randn(1, 3, 1024, 1024).clamp(-1, 1)
print(f"  Dummy input shape: {dummy_image.shape}")

# 4. Test PyTorch Inference
print("Testing PyTorch inference...")
with torch.no_grad():
    test_output = wrapper(dummy_image)
print(f"  Output shape: {test_output.shape}")  # [1, 4, 128, 128]

# 5. TorchScript Trace
print("Tracing model...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (dummy_image,), strict=False)
traced = torch.jit.freeze(traced)
print("  Trace successful!")

# 6. Convert to CoreML
print("Converting to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="image", shape=(1, 3, 1024, 1024))],
    outputs=[ct.TensorType(name="latent")],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.ALL,
)

# 7. 保存
save_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
mlmodel.save(str(save_path))
size_mb = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file()) / 1024**2
print(f"\n✅ VAE Encoder exported successfully!")
print(f"   Path: {save_path}")
print(f"   Size: {size_mb:.1f} MB")
print(f"   Input:  image  [1, 3, 1024, 1024]  (normalized to [-1, 1])")
print(f"   Output: latent [1, 4, 128, 128]")