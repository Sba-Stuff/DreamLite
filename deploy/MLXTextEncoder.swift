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

/// Mode determines which prompt template is used and how many leading tokens are
/// dropped from the encoder hidden states.
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

/// Wraps a 4-bit MLX-quantised Qwen3-VL and exposes a `(hidden_states, mask)`
/// API that downstream CoreML modules can consume directly.
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

    /// Convenience initializer that locates `config.json` inside the app bundle.
    convenience init?() {
        if let configURL = Bundle.main.url(
            forResource: "config", withExtension: "json", subdirectory: "Models/text_encoder_mlx_4bit"
        ) {
            self.init(modelDirectory: configURL.deletingLastPathComponent())
            return
        }
        if let configURL = Bundle.main.url(forResource: "config", withExtension: "json") {
            self.init(modelDirectory: configURL.deletingLastPathComponent())
            return
        }
        print("[MLXTextEncoder] config.json not found in bundle")
        return nil
    }

    // MARK: - Load model

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

        // Merge all safetensors shards.
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

        // Identify modules that have quantized weights (path.scales exists).
        var quantizedPaths = Set<String>()
        for key in sanitized.keys where key.hasSuffix(".scales") {
            quantizedPaths.insert(String(key.dropLast(".scales".count)))
        }

        // Build replacements: convert Linear → QuantizedLinear, Embedding → QuantizedEmbedding.
        let leaves = vlm.leafModules().flattened()
        var replacements: [(String, Module)] = []
        var handledKeys = Set<String>()

        for (path, module) in leaves {
            guard quantizedPaths.contains(path) else { continue }

            guard let qWeight = sanitized[path + ".weight"],
                  let scales = sanitized[path + ".scales"] else { continue }
            let biases = sanitized[path + ".biases"]

            if module is Linear, !(module is QuantizedLinear) {
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
                // Dequantize → re-quantize via QuantizedEmbedding constructor.
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
        vlm.update(modules: ModuleChildren.unflattened(replacements))

        // Load remaining (non-quantized) weights, e.g. RMSNorm, layernorm.
        var remainingWeights: [String: MLXArray] = [:]
        for (key, value) in sanitized where !handledKeys.contains(key) {
            remainingWeights[key] = value
        }
        let params = ModuleParameters.unflattened(remainingWeights)
        try vlm.update(parameters: params, verify: .none)

        tokenizer = try await AutoTokenizer.from(modelFolder: modelPath)
        MLX.GPU.set(cacheLimit: 512 * 1024 * 1024)
        model = vlm
        isLoaded = true

        print("[MLXTextEncoder] Ready in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t0))s")
    }

    // MARK: - Encode (text-only)

    func encodePrompt(
        prompt: String,
        mode: PromptMode = .generate,
        maxLength: Int = 100
    ) throws -> (hiddenStates: MLMultiArray, attentionMask: MLMultiArray) {

        guard let model = model, let tokenizer = tokenizer else {
            throw MLXEncoderError.modelNotLoaded
        }

        let fullPrompt = buildPrompt(prompt: prompt, mode: mode)
        let tokens = tokenizer.encode(text: fullPrompt)
        let seqLen = tokens.count

        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, seqLen])
        let hidden = model.getHiddenStates(inputIds: inputIds)
        eval(hidden)

        let dropIdx = mode.dropIdx
        guard seqLen > dropIdx else {
            throw MLXEncoderError.sequenceTooShort(seqLen, dropIdx)
        }

        let trimmed = hidden[0..., dropIdx..., 0...]
        let trimmedLen = seqLen - dropIdx
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

    // MARK: - Encode (edit: text + image)

    func encodeEditPrompt(
        instruction: String,
        sourceImage: UIImage,
        maxLength: Int = 200
    ) throws -> (hiddenStates: MLMultiArray, attentionMask: MLMultiArray) {

        guard let model = model, let tokenizer = tokenizer else {
            throw MLXEncoderError.modelNotLoaded
        }

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

    // MARK: - Image preprocessing

    /// Qwen3-VL processor: resize → 256×256, patch_size=16, temporal_patch_size=2.
    /// Output `(pixelValues [num_patches, 1536], gridTHW [1, 16, 16])`.
    private func preprocessImageForVision(_ image: UIImage) throws -> (MLXArray, [THW]) {
        let patchSize = 16
        let temporalPatchSize = 2
        let actualSize = 256
        let gridH = actualSize / patchSize    // 16
        let gridW = actualSize / patchSize    // 16
        let numPatches = gridH * gridW

        // Resize to 256×256.
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

        // CLIP normalization values (used by Qwen3-VL processor).
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        let std:  [Float] = [0.26862954, 0.26130258, 0.27577711]

        // patch_dim = temporal_patch_size · 3 · patch_size² = 1536.
        let patchDim = temporalPatchSize * 3 * patchSize * patchSize
        var patches = [Float](repeating: 0, count: numPatches * patchDim)

        for ph in 0..<gridH {
            for pw in 0..<gridW {
                let patchIdx = ph * gridW + pw
                // Single-frame image is duplicated across the 2 temporal slots.
                for t in 0..<temporalPatchSize {
                    for c in 0..<3 {
                        for py in 0..<patchSize {
                            for px in 0..<patchSize {
                                let imgY = ph * patchSize + py
                                let imgX = pw * patchSize + px
                                let pixelIdx = (imgY * w + imgX) * 4
                                let val = Float(rawPixels[pixelIdx + c]) / 255.0
                                let normalized = (val - mean[c]) / std[c]

                                let offsetInPatch =
                                    t * (3 * patchSize * patchSize)
                                  + c * (patchSize * patchSize)
                                  + py * patchSize + px
                                patches[patchIdx * patchDim + offsetInPatch] = normalized
                            }
                        }
                    }
                }
            }
        }

        let pixelValues = MLXArray(patches, [numPatches, patchDim])
        return (pixelValues, [THW(1, gridH, gridW)])
    }

    // MARK: - Prompt templates

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

    // MARK: - MLX → MLMultiArray

    private func toMLMultiArray(_ array: MLXArray, shape: [Int]) throws -> MLMultiArray {
        let f32 = array.asType(.float32)
        eval(f32)
        let nsShape = shape.map { NSNumber(value: $0) }
        let ml = try MLMultiArray(shape: nsShape, dataType: .float16)
        let total = shape.reduce(1, *)
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
        let ptr = ml.dataPointer.bindMemory(to: Float16.self, capacity: values.count)
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
        case .modelNotLoaded:                return "Text encoder not loaded."
        case .sequenceTooShort(let s, let d): return "Sequence (\(s)) shorter than drop index (\(d))."
        case .conversionFailed(let m):        return m
        case .preprocessingFailed:            return "Preprocessing failed."
        }
    }
}
