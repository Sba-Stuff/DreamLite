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

import torch
from safetensors.torch import load_file
from dreamlite import DreamLitePipelineLoRA
from diffusers.utils import load_image
from peft import PeftModel

pipe = DreamLitePipelineLoRA.from_pretrained("models/DreamLite-base", torch_dtype=torch.bfloat16).to("cuda")

# 1. Load LoRA Weights
lora_path = "output/output_lora/yarn"
# input_image = load_image("output/a_photo_of_a_cat.png")  # for Edit lora

print(f"Injecting LoRA weights from {lora_path}...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

# 3. Inference
image = pipe(
    prompt="A girl in the forest, yarn art style", 
    # image=input_image,
    num_inference_steps=28,
    image_guidance_scale=1.5
).images[0]

image.save("output/yarn_lora.png")
