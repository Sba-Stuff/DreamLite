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

import CoreML
import Foundation

// MARK: - UNet Inference

class UNetModel {
    let model: MLModel
    
    init() throws {
        // Load unet.mlpackage from app bundle
        guard let modelURL = Bundle.main.url(forResource: "unet", withExtension: "mlmodelc") ??
              Bundle.main.url(forResource: "unet", withExtension: "mlpackage") else {
            throw DreamLiteError.modelNotFound("unet")
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use CPU + GPU 这是最快的
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }
    
    /// Run UNet inference
    /// - Parameters:
    ///   - sample: [1, 4, 128, 256] noisy latent (in-context concatenated)
    ///   - timestep: [1] current timestep value
    ///   - encoderHiddenStates: [1, N, 2048] text embeddings from Qwen3-VL
    ///   - encoderAttentionMask: [1, N] attention mask
    ///   - timeIds: [1, 2] width & height
    /// - Returns: [1, 4, 128, 256] predicted noise
    func predict(
        sample: MLMultiArray,
        timestep: MLMultiArray,
        encoderHiddenStates: MLMultiArray,
        encoderAttentionMask: MLMultiArray,
        timeIds: MLMultiArray
    ) throws -> MLMultiArray {
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "sample": MLFeatureValue(multiArray: sample),
            "timestep": MLFeatureValue(multiArray: timestep),
            "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
            "encoder_attention_mask": MLFeatureValue(multiArray: encoderAttentionMask),
            "time_ids": MLFeatureValue(multiArray: timeIds),
        ])
        
        let output = try model.prediction(from: inputFeatures)
        guard let noisePred = output.featureValue(for: "noise_pred")?.multiArrayValue else {
            throw DreamLiteError.inferenceError("UNet output missing 'noise_pred'")
        }
        return noisePred
    }
}

// MARK: - VAE Decoder

class VAEDecoder {
    let model: MLModel
    
    init() throws {
        guard let modelURL = Bundle.main.url(forResource: "vae_decoder", withExtension: "mlmodelc") ??
              Bundle.main.url(forResource: "vae_decoder", withExtension: "mlpackage") else {
            throw DreamLiteError.modelNotFound("vae_decoder")
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }
    
    /// Decode latent to image
    /// - Parameter latent: [1, 4, 128, 128] latent representation
    /// - Returns: [1, 3, 1024, 1024] decoded image
    func decode(latent: MLMultiArray) throws -> MLMultiArray {
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "latent": MLFeatureValue(multiArray: latent),
        ])
        
        let output = try model.prediction(from: inputFeatures)
        guard let image = output.featureValue(for: "image")?.multiArrayValue else {
            throw DreamLiteError.inferenceError("VAE output missing 'image'")
        }
        return image
    }
}

// MARK: - Error Types

enum DreamLiteError: LocalizedError {
    case modelNotFound(String)
    case inferenceError(String)
    case invalidInput(String)
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model '\(name)' not found in app bundle"
        case .inferenceError(let msg):
            return "Inference error: \(msg)"
        case .invalidInput(let msg):
            return "Invalid input: \(msg)"
        }
    }
}