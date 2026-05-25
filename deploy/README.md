# DreamLite Mobile Deployment Guide

This directory contains all the code needed to export models and deploy DreamLite on iOS (Swift/CoreML).

## Directory Structure

```
deploy/
├── export_unet.py              # Export UNet to CoreML (fp16)
├── export_vae_decoder.py               # Export VAE decoder to CoreML (fp16)
├── export_vae_encoder.py               # Export VAE encoder to CoreML (fp16)
├── export_text_encoder.py      # Export Qwen3-VL text encoder (two-step process)
├── Qwen3VL.swift               # Modified mlx-swift-lm file (for DreamLite v1.0)
├── Qwen35.swift                # Modified mlx-swift-lm file (for DreamLite v1.1+)
├── ContentView.swift           # iOS UI interface
├── CoreMLModels.swift          # CoreML model loading (UNet & VAE)
├── DreamLiteApp.swift          # App entry point (main)
├── DreamLitePipeline.swift     # Inference pipeline
├── FluxScheduler.swift         # Flow matching scheduler (Swift)
└── MLXTextEncoder.swift        # Text encoder loading via MLX
```

## 1. Model Export

### UNet & VAE

Export UNet and VAE (encoder + decoder) to CoreML format with **fp16** precision:

```bash
python export_unet.py
python export_vae.py
```

### Text Encoder (Two-Step Process)

The text encoder export requires **two steps**: first export to safetensors via Python, then quantize to 4-bit using `mlx-vlm` on **macOS**.

**Step 1**: Export safetensors (can run on Linux/macOS)

```bash
python export_text_encoder.py
```

**Step 2**: Quantize with mlx-vlm (macOS only)

```bash
# Install mlx-vlm if not already installed
pip install mlx-vlm

# Quantize to 4-bit (command is printed at the end of Step 1)
python -m mlx_vlm.convert --hf-path <model_path> --mlx-path <output_path> -q --q-bits 4
```

## 2. mlx-swift-lm Library Modification

Since `mlx-swift-lm` does not natively support extracting the last hidden states from the text encoder or injecting image tokens, you need to modify the library source code.

### Setup

1. Clone `mlx-swift-lm` to your workspace
2. Copy the modified files from this directory:

```bash
# For DreamLite v1.0 (Qwen3-VL based)
cp deploy/Qwen3VL.swift  <your-mlx-swift-lm>/Libraries/MLXVLM/Models/Qwen3VL.swift

# For DreamLite v1.1+ (Qwen3.5 based)
cp deploy/Qwen35.swift   <your-mlx-swift-lm>/Libraries/MLXVLM/Models/Qwen35.swift
```

### Key Modifications

- Removed `private` access modifier from internal methods (`applyDeepstack`, `mergeInputIdsWithImageFeatures`, `cumulativeSplitIndices`)
- Added `getHiddenStates(inputIds:)` — extract hidden states for text-only input
- Added `getHiddenStatesWithImage(inputIds:pixelValues:imageGridTHW:)` — extract hidden states for text+image input (used in editing mode)

## 3. iOS App Structure

Place the Swift files in your Xcode project following this structure:

```
DreamLite/
├── DreamLite/
│   ├── Models/
│   │   ├── text_encoder_mlx_4bit/   # 4-bit quantized text encoder
│   │   ├── unet/                     # CoreML UNet model
│   │   ├── vae_decoder/              # CoreML VAE decoder
│   │   └── vae_encoder/              # CoreML VAE encoder
│   ├── ContentView.swift             # UI interface
│   ├── CoreMLModels.swift            # Load UNet & VAE
│   ├── DreamLiteApp.swift            # main()
│   ├── DreamLitePipeline.swift       # Inference pipeline
│   ├── FluxScheduler.swift           # Flow matching scheduler
│   └── MLXTextEncoder.swift          # Load text encoder
├── DreamLiteTests/
└── DreamLiteUITests/
```

### Model Placement

Export all models into the `Models/` directory before building:
- `text_encoder_mlx_4bit/` — output from Step 2 of text encoder export
- `unet/` — CoreML package from `export_unet.py`
- `vae_decoder/` and `vae_encoder/` — CoreML packages from `export_vae.py`

## 4. Requirements

| Component | Requirement |
|-----------|-------------|
| Model Export (UNet/VAE) | Python 3.10+, coremltools, PyTorch |
| Text Encoder Quantization | macOS, mlx-vlm |
| iOS Development | Xcode 15+, iOS 17+, mlx-swift, mlx-swift-lm |
| Device | iPhone 16 Pro / Pro Max or later (8GB+ RAM) |

## 5. Xcode Project File Mapping

Below shows how files in this `deploy/` directory map to the Xcode project structure:

```
Xcode Workspace
├── dreamlite-mlx-swift-lm/                          # mlx-swift-lm (Swift Package)
│   └── Libraries/
│       └── MLXVLM/
│           └── Models/
│               ├── Qwen3VL.swift                    ← deploy/Qwen3VL.swift
│               └── Qwen35.swift                     ← deploy/Qwen35.swift
│
└── DreamLite/                                        # iOS App Target
    ├── DreamLite/
    │   ├── Models/                                   # Bundle Resources (exported models)
    │   │   ├── text_encoder_mlx_4bit/
    │   │   ├── unet.mlpackage/
    │   │   ├── vae_decoder.mlpackage/
    │   │   └── vae_encoder.mlpackage/
    │   ├── ContentView.swift                        ← deploy/ContentView.swift
    │   ├── CoreMLModels.swift                       ← deploy/CoreMLModels.swift
    │   ├── DreamLiteApp.swift                       ← deploy/DreamLiteApp.swift
    │   ├── DreamLitePipeline.swift                  ← deploy/DreamLitePipeline.swift
    │   ├── FluxScheduler.swift                      ← deploy/FluxScheduler.swift
    │   └── MLXTextEncoder.swift                     ← deploy/MLXTextEncoder.swift
    ├── DreamLiteTests/
    └── DreamLiteUITests/
```

### Xcode Setup Steps

1. Open/create the Xcode workspace
2. Add `dreamlite-mlx-swift-lm` as a local Swift Package dependency (with modified `Qwen3VL.swift` / `Qwen35.swift`)
3. Place all `.swift` source files under `DreamLite/DreamLite/`
4. Add exported model folders to `DreamLite/DreamLite/Models/` and ensure they are included in **Copy Bundle Resources** build phase
5. In **Build Settings**, link the `MLXVLM` library from the local Swift Package

## Notes

- The exported UNet and VAE use **fp16** precision for optimal on-device performance.
- The text encoder uses **4-bit quantization** to reduce memory footprint (~0.8GB).
- Make sure to use the modified `mlx-swift-vlm` library, not the original version.