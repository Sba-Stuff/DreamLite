// Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import CoreML
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import Tokenizers
import UIKit

enum PromptMode {
    case generate
    case edit
    
    var dropIdx: Int {
        switch self {
        case .generate: return 34
        case .edit:     return 64
        }
    }
}

class MLXTextEncoder {
    
    private var model: Qwen3VL?
    private var tokenizer: (any Tokenizer)?
    private var isLoaded = false
    private let hiddenSize = 2048
    private let modelPath: URL
    
    private let systemPrompt =
        "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
    
    init(modelDirectory: URL) {
        self.modelPath = modelDirectory
    }
    
    convenience init?() {
        // text_encoder 文件在 bundle 里是平铺的，不在子目录中
        // 通过查找 config.json 来定位目录
        guard let configURL = Bundle.main.url(
            forResource: "config", withExtension: "json", subdirectory: "Models/text_encoder_mlx_4bit"
        ) else {
            // 尝试不带子目录（文件可能在 bundle 根目录）
            guard let configURL = Bundle.main.url(
                forResource: "config", withExtension: "json"
            ) else {
                print("[MLXTextEncoder] config.json not found in bundle")
                return nil
            }
            self.init(modelDirectory: configURL.deletingLastPathComponent())
            return
        }
        self.init(modelDirectory: configURL.deletingLastPathComponent())
    }
    
    // MARK: - 加载模型
    
    func loadModel() async throws {
        guard !isLoaded else { return }
        let t0 = CFAbsoluteTimeGetCurrent()
        print("[MLXTextEncoder] Loading from \(modelPath.path)")
        
        let configData = try Data(contentsOf: modelPath.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: configData)
        let vlm = Qwen3VL(config)
        
        var quantBits = 4
        var quantGroupSize = 64
        if let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let qConfig = configDict["quantization"] as? [String: Any] {
            quantBits = qConfig["bits"] as? Int ?? 4
            quantGroupSize = qConfig["group_size"] as? Int ?? 64
        }
        print("[MLXTextEncoder] Quantization: \(quantBits)-bit, group_size=\(quantGroupSize)")
        
        // 合并所有 safetensors
        let files = try FileManager.default.contentsOfDirectory(
            at: modelPath, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }
         .sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        var allWeights: [String: MLXArray] = [:]
        for file in files {
            let raw = try MLX.loadArrays(url: file)
            for (k, v) in raw { allWeights[k] = v }
        }
        
        let sanitized = vlm.sanitize(weights: allWeights)
        
        // ⭐ 找出所有量化路径
        var quantizedPaths = Set<String>()
        for key in sanitized.keys where key.hasSuffix(".scales") {
            quantizedPaths.insert(String(key.dropLast(".scales".count)))
        }
        print("[MLXTextEncoder] Found \(quantizedPaths.count) quantized modules")
        
        // ⭐ 手动构建量化模块并替换
        let leaves = vlm.leafModules().flattened()
        var replacements: [(String, Module)] = []
        var handledKeys = Set<String>()
        
        for (path, module) in leaves {
            guard quantizedPaths.contains(path) else { continue }
            
            guard let qWeight = sanitized[path + ".weight"],
                  let scales = sanitized[path + ".scales"] else { continue }
            let biases = sanitized[path + ".biases"]
            
            if module is Linear, !(module is QuantizedLinear) {
                // Linear → QuantizedLinear（直接传入已量化的权重）
                let bias = sanitized[path + ".bias"]
                let ql = QuantizedLinear(
                    weight: qWeight, bias: bias,
                    scales: scales, biases: biases,
                    groupSize: quantGroupSize, bits: quantBits
                )
                replacements.append((path, ql))
                handledKeys.insert(path + ".weight")
                handledKeys.insert(path + ".scales")
                handledKeys.insert(path + ".biases")
                if bias != nil { handledKeys.insert(path + ".bias") }
                
            } else if module is Embedding, !(module is QuantizedEmbedding) {
                // Embedding → 先 dequantize 再让 QuantizedEmbedding 重新量化
                let dqWeight = dequantized(
                    qWeight, scales: scales, biases: biases,
                    groupSize: quantGroupSize, bits: quantBits
                )
                eval(dqWeight)
                let qe = QuantizedEmbedding(
                    weight: dqWeight,
                    groupSize: quantGroupSize, bits: quantBits
                )
                replacements.append((path, qe))
                handledKeys.insert(path + ".weight")
                handledKeys.insert(path + ".scales")
                handledKeys.insert(path + ".biases")
            }
        }
        
        print("[MLXTextEncoder] Replacing \(replacements.count) modules")
        print("[DEBUG] Total replacements: \(replacements.count)")
        for (path, module) in replacements.prefix(5) {
            print("[DEBUG] Replacement: \(path) → \(type(of: module))")
        }
        vlm.update(modules: ModuleChildren.unflattened(replacements))
        
        // ⭐ 加载剩余非量化权重（RMSNorm 等）
        var remainingWeights: [String: MLXArray] = [:]
        for (key, value) in sanitized {
            if !handledKeys.contains(key) {
                remainingWeights[key] = value
            }
        }
        print("[MLXTextEncoder] Loading \(remainingWeights.count) remaining weights")
        let params = ModuleParameters.unflattened(remainingWeights)
        try vlm.update(parameters: params, verify: .none)
        
        // tokenizer
        tokenizer = try await AutoTokenizer.from(modelFolder: modelPath)
        MLX.GPU.set(cacheLimit: 512 * 1024 * 1024)
        model = vlm
        
        var linearCount = 0
        var quantLinearCount = 0
        var embeddingCount = 0
        var quantEmbeddingCount = 0

        for (path, module) in vlm.leafModules().flattened() {
            if module is QuantizedLinear {
                quantLinearCount += 1
            } else if module is Linear {
                linearCount += 1
            }
            if module is QuantizedEmbedding {
                quantEmbeddingCount += 1
            } else if module is Embedding {
                embeddingCount += 1
            }
        }
        print("[DEBUG] Module counts: Linear=\(linearCount), QuantizedLinear=\(quantLinearCount), Embedding=\(embeddingCount), QuantizedEmbedding=\(quantEmbeddingCount)")

        for (i, (path, module)) in vlm.leafModules().flattened().prefix(10).enumerated() {
            print("[DEBUG] leaf[\(i)] \(path) → \(type(of: module))")
        }
        
        isLoaded = true
        
        // 验证
        for (path, module) in vlm.leafModules().flattened() {
            if path.contains("layers.0.self_attn.q_proj"), let ql = module as? QuantizedLinear {
                eval(ql.scales)
                print("[DEBUG] \(path) type=QuantizedLinear, scales range: [\(ql.scales.min().item(Float.self)), \(ql.scales.max().item(Float.self))]")
                break
            }
        }
        
        print("[MLXTextEncoder] Ready in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t0))s")
    }
    
    // MARK: - 编码 Prompt
    
    func encodePrompt(
        prompt: String,
        mode: PromptMode = .generate,
        maxLength: Int = 100
    ) throws -> (hiddenStates: MLMultiArray, attentionMask: MLMultiArray) {
        
        guard let model = model, let tokenizer = tokenizer else {
            throw MLXEncoderError.modelNotLoaded
        }
        
        let fullPrompt = buildPrompt(prompt: prompt, mode: mode)
        print("[MLXTextEncoder] Prompt: \(fullPrompt.prefix(80))...")
        
        let tokens = tokenizer.encode(text: fullPrompt)
        print("[DEBUG] edit prompt token count: \(tokens.count)")
        print("[DEBUG] fullPrompt prefix: \(String(fullPrompt.prefix(200)))")
        let seqLen = tokens.count
        print("[MLXTextEncoder] Tokens: \(seqLen)")
        
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, seqLen])
        let hidden = model.getHiddenStates(inputIds: inputIds)
        eval(hidden)
        
        // ---- 调试：打印 hidden states 的详细统计 ----
        let hiddenFlat = hidden.reshaped([-1])
        eval(hiddenFlat)
        let allVals = hiddenFlat.asArray(Float.self)
        let mean = allVals.reduce(0, +) / Float(allVals.count)
        let absMax = allVals.map { abs($0) }.max() ?? 0
        let nanCount = allVals.filter { $0.isNaN }.count
        let infCount = allVals.filter { $0.isInfinite }.count
        print("[MLXTextEncoder] Hidden stats: mean=\(mean), absMax=\(absMax), nan=\(nanCount), inf=\(infCount), shape=\(hidden.shape)")

        // 也打印 trimmed 后的统计
        
        let dropIdx = mode.dropIdx
        guard seqLen > dropIdx else {
            throw MLXEncoderError.sequenceTooShort(seqLen, dropIdx)
        }
        let trimmed = hidden[0..., dropIdx..., 0...]
        let trimmedLen = seqLen - dropIdx
        print("[MLXTextEncoder] After drop: \(seqLen) → \(trimmedLen)")
        
        let actualLen = min(trimmedLen, maxLength)
        let padded: MLXArray
        if trimmedLen >= maxLength {
            padded = trimmed[0..., ..<maxLength, 0...]
        } else {
            let zeros = MLXArray.zeros([1, maxLength - trimmedLen, hiddenSize])
            padded = concatenated([trimmed, zeros], axis: 1)
        }
        
        var maskVals = [Float](repeating: 1.0, count: actualLen)
        if actualLen < maxLength {
            maskVals += [Float](repeating: 0.0, count: maxLength - actualLen)
        }
        
        eval(padded)
        let paddedFlat = padded.reshaped([-1])
        eval(paddedFlat)
        let pVals = paddedFlat.asArray(Float.self)
        let pAbsMax = pVals.map { abs($0) }.max() ?? 0
        let pNanCount = pVals.filter { $0.isNaN }.count
        print("[MLXTextEncoder] Padded stats: absMax=\(pAbsMax), nan=\(pNanCount)")
        
        let lo = padded.min().item(Float.self)
        let hi = padded.max().item(Float.self)
        print("[MLXTextEncoder] Hidden range: [\(lo), \(hi)]")
        
        let hiddenML = try toMLMultiArray(padded, shape: [1, maxLength, hiddenSize])
        let maskML = try toMLMultiArrayFromFloats(maskVals, shape: [1, maxLength])
        
        return (hiddenML, maskML)
    }
    
    /// 编辑模式：文本 + 图像 → hidden states
    // MLXTextEncoder.swift — 改签名，接收 UIImage 而不是 MLXArray
    func encodeEditPrompt(
        instruction: String,
        sourceImage: UIImage,
        maxLength: Int = 200
    ) throws -> (hiddenStates: MLMultiArray, attentionMask: MLMultiArray) {
        print("[DEBUG] encodeEditPrompt called, instruction length: \(instruction.count)")
        fflush(stdout)
        
        guard let model = model, let tokenizer = tokenizer else {
            throw MLXEncoderError.modelNotLoaded
        }

        // 预处理图片 → MLXArray (224×224, CLIP normalization)
        let (pixelValues, imageGridTHW) = try preprocessImageForVision(sourceImage)

        let mode = PromptMode.edit
        let fullPrompt = buildPrompt(prompt: instruction, mode: mode)
        let tokens = tokenizer.encode(text: fullPrompt)
        let seqLen = tokens.count
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, seqLen])

        let hidden = model.getHiddenStatesWithImage(
            inputIds: inputIds, pixelValues: pixelValues, imageGridTHW: imageGridTHW
        )
        eval(hidden)

        // ... 后面 trim/pad 逻辑不变 ...
        let totalSeqLen = hidden.shape[1]
        let dropIdx = mode.dropIdx
        guard totalSeqLen > dropIdx else {
            throw MLXEncoderError.sequenceTooShort(totalSeqLen, dropIdx)
        }
        let trimmed = hidden[0..., dropIdx..., 0...]
        let trimmedLen = totalSeqLen - dropIdx
        let actualLen = min(trimmedLen, maxLength)
        let padded: MLXArray
        if trimmedLen >= maxLength {
            padded = trimmed[0..., ..<maxLength, 0...]
        } else {
            let zeros = MLXArray.zeros([1, maxLength - trimmedLen, hiddenSize])
            padded = concatenated([trimmed, zeros], axis: 1)
        }
        var maskVals = [Float](repeating: 1.0, count: actualLen)
        if actualLen < maxLength {
            maskVals += [Float](repeating: 0.0, count: maxLength - actualLen)
        }
        eval(padded)
        let hiddenML = try toMLMultiArray(padded, shape: [1, maxLength, hiddenSize])
        let maskML = try toMLMultiArrayFromFloats(maskVals, shape: [1, maxLength])
        return (hiddenML, maskML)
    }

    // 在 MLXTextEncoder.swift 里加这个私有方法
    private func preprocessImageForVision(_ image: UIImage) throws -> (MLXArray, [THW]) {
        let targetSize = 224  // processor resize 到 224×224
        let patchSize = 16    // config: patch_size = 16
        let temporalPatchSize = 2  // config: temporal_patch_size = 2
        let gridH = targetSize / patchSize  // 14... 不对，实际是 16

        // 实际 processor 做了 resize 使得 grid 恰好是 16×16
        // 16 × 16 patches × patch_size=16 = 256×256 的图
        // 所以 processor 实际 resize 到 256×256，不是 224×224
        let actualSize = 256
        let gridH2 = actualSize / patchSize  // 16
        let gridW2 = actualSize / patchSize  // 16
        let numPatches = gridH2 * gridW2     // 256

        // Resize to 256×256
        let size = CGSize(width: actualSize, height: actualSize)
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        guard let cgImage = resized.cgImage else {
            throw MLXEncoderError.preprocessingFailed
        }
        let w = cgImage.width
        let h = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawPixels = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &rawPixels, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw MLXEncoderError.preprocessingFailed
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        // Normalize to [0, 1] then rescale to ~[-1, 1] range
        // Python processor uses rescale_factor=1/255 then normalize with ImageNet stats
        // But actual output range is [-0.97, 0.99], which matches simple /255 normalization
        // Qwen3-VL processor: rescale(1/255) → normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711])
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        let std: [Float]  = [0.26862954, 0.26130258, 0.27577711]

        // Build patches: [num_patches, patch_dim]
        // patch_dim = temporal_patch_size * patchSize * patchSize * 3 = 2 * 16 * 16 * 3 = 1536
        // But we only have 1 frame, temporal_patch_size=2 means we duplicate the frame
        let patchDim = temporalPatchSize * patchSize * patchSize * 3  // 1536
        var patches = [Float](repeating: 0, count: numPatches * patchDim)

        for ph in 0..<gridH2 {
            for pw in 0..<gridW2 {
                let patchIdx = ph * gridW2 + pw
                // For each temporal frame (duplicate since we have 1 image, temporal_patch_size=2)
                for t in 0..<temporalPatchSize {
                    for c in 0..<3 {
                        for py in 0..<patchSize {
                            for px in 0..<patchSize {
                                let imgY = ph * patchSize + py
                                let imgX = pw * patchSize + px
                                let pixelIdx = (imgY * w + imgX) * 4
                                let val = Float(rawPixels[pixelIdx + c]) / 255.0
                                let normalized = (val - mean[c]) / std[c]

                                // patch layout: [t, c, py, px] flattened
                                let offsetInPatch = t * (3 * patchSize * patchSize) + c * (patchSize * patchSize) + py * patchSize + px
                                patches[patchIdx * patchDim + offsetInPatch] = normalized
                            }
                        }
                    }
                }
            }
        }

        let pixelValues = MLXArray(patches, [numPatches, patchDim])

        print("[DEBUG] pixelValues shape: \(pixelValues.shape), min: \(pixelValues.min()), max: \(pixelValues.max())")

        let grid = [THW(1, gridH2, gridW2)]  // [1, 16, 16]
        return (pixelValues, grid)
    }

    /// 图像预处理：UIImage → (pixelValues, gridTHW)
    private func preprocessImage(_ image: UIImage) -> (MLXArray, [THW]) {
        let targetSize = CGSize(width: 224, height: 224)
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        let resized = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        guard let cgImage = resized.cgImage else {
            fatalError("Failed to get CGImage")
        }
        let width = 224
        let height = 224
        var rawPixels = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &rawPixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // ImageNet 归一化：Qwen3-VL 用的是 CLIP 预处理
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        let std: [Float] = [0.26862954, 0.26130258, 0.27577711]
        
        var floatPixels = [Float](repeating: 0, count: 3 * height * width)
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = (y * width + x) * 4
                for c in 0..<3 {
                    let val = Float(rawPixels[srcIdx + c]) / 255.0
                    let dstIdx = c * height * width + y * width + x
                    floatPixels[dstIdx] = (val - mean[c]) / std[c]
                }
            }
        }
        
        let pixelValues = MLXArray(floatPixels, [1, 3, height, width])
        
        // Qwen3-VL grid_thw: 单张 224x224，patch_size=14
        // t=1, h=224/14=16, w=224/14=16
        let gridTHW: [THW] = [THW(1, 16, 16)]
        
        return (pixelValues, gridTHW)
    }
    
    // MARK: - Prompt 模板
    
    private func buildPrompt(prompt: String, mode: PromptMode) -> String {
        switch mode {
        case .generate:
            return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        case .edit:
            let editSystemPrompt = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
            let visionTokenCount = 64
            let imagePads = String(repeating: "<|image_pad|>", count: visionTokenCount)
            return "<|im_start|>system\n\(editSystemPrompt)<|im_end|>\n<|im_start|>user\n<|vision_start|>\(imagePads)<|vision_end|>\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        }
    }
    
    // MARK: - 类型转换
    
    private func toMLMultiArray(_ array: MLXArray, shape: [Int]) throws -> MLMultiArray {
        let f32 = array.asType(.float32)
        eval(f32)
        let nsShape = shape.map { NSNumber(value: $0) }
        let ml = try MLMultiArray(shape: nsShape, dataType: .float16)
        let total = shape.reduce(1, *)
        
        // MLXArray → [Float] → MLMultiArray
        let floatValues = f32.asArray(Float.self)
        
        let ptr = ml.dataPointer.bindMemory(to: Float16.self, capacity: total)
        for i in 0..<total {
            ptr[i] = Float16(floatValues[i])
        }
        return ml
    }
    
    private func toMLMultiArrayFromFloats(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let ml = try MLMultiArray(shape: nsShape, dataType: .float16)
        let total = shape.reduce(1, *)
        
        let ptr = ml.dataPointer.bindMemory(to: Float16.self, capacity: total)
        for i in 0..<values.count {
            ptr[i] = Float16(values[i])
        }
        return ml
    }
}

enum MLXEncoderError: LocalizedError {
    case modelNotLoaded
    case sequenceTooShort(Int, Int)
    case conversionFailed(String)
    case preprocessingFailed
    
    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Text encoder not loaded."
        case .sequenceTooShort(let s, let d): return "Seq(\(s)) < drop(\(d))"
        case .conversionFailed(let m): return m
        case .preprocessingFailed: return "Preprocessing Failed"
        }
    }
}

