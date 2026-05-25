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
import Accelerate
import UIKit
import Foundation

func printMemoryUsage(_ tag: String) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    
    let result: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    
    if result == KERN_SUCCESS {
        let memoryMB = Double(info.phys_footprint) / 1024 / 1024
        print("📊 [\(tag)] 总内存: \(String(format: "%.1f", memoryMB))MB")
    } else {
        print("📊 [\(tag)] 获取内存失败: \(result)")
    }
}

/// Main DreamLite inference pipeline
/// Orchestrates: Text Encoder → UNet denoising loop → VAE decode → Image
class DreamLitePipeline {
    
    let unet: UNetModel
    let vae: VAEDecoder
    let scheduler: FluxScheduler
    
    // Model dimensions
    let latentChannels = 4
    let latentHeight = 128       // 1024 / 8
    let latentWidth = 128        // 1024 / 8
    let inContextWidth = 256     // latentWidth * 2 (noise + cond_image concatenated)
    let hiddenSize = 2048        // Qwen3-VL output dim
    
    init() throws {
        printMemoryUsage("Pipeline初始化前")
        print("[Pipeline] Loading models...")
        self.unet = try UNetModel()
        printMemoryUsage("UNet加载完成")
        print("[Pipeline] UNet loaded ✓")
        self.vae = try VAEDecoder()
        printMemoryUsage("VAE加载完成")
        print("[Pipeline] VAE loaded ✓")
        self.scheduler = FluxScheduler()
        printMemoryUsage("所有模型加载完成")
        print("[Pipeline] Scheduler ready ✓")
    }
    
    // MARK: - Generate Image (text-only mode)
    
    /// Generate an image from text prompt
    /// - Parameters:
    ///   - promptEmbeds: [1, N, 2048] text embeddings from Qwen3-VL
    ///   - attentionMask: [1, N] attention mask
    ///   - numInferenceSteps: number of denoising steps (default: 4)
    ///   - width: output image width (default: 1024)
    ///   - height: output image height (default: 1024)
    ///   - progressHandler: callback for progress updates
    /// - Returns: [1, 3, 1024, 1024] decoded image as MLMultiArray
    func generate(
        promptEmbeds: MLMultiArray,
        attentionMask: MLMultiArray,
        numInferenceSteps: Int = 4,
        width: Int = 1024,
        height: Int = 1024,
        conditionLatent: MLMultiArray? = nil,
        progressHandler: ((Int, Int) -> Void)? = nil
    ) throws -> MLMultiArray {
        printMemoryUsage("推理开始前")
        // 1. Prepare sigmas (matching Python: np.linspace(1.0, 1/N, N))
        var sigmas: [Double] = []
        for i in 0..<numInferenceSteps {
            let step = Double(i)
            let total = Double(numInferenceSteps)
            let value = 1.0 - step * (1.0 - 1.0 / total) / (total - 1.0)
            sigmas.append(value)
        }
        
        // 2. Calculate mu for dynamic shifting
        let imageSeqLen = (latentHeight * latentWidth) / 4
        let mu = FluxScheduler.calculateShift(
            imageSeqLen: imageSeqLen,
            baseSeqLen: scheduler.baseImageSeqLen,
            maxSeqLen: scheduler.maxImageSeqLen,
            baseShift: scheduler.baseShift,
            maxShift: scheduler.maxShift
        )
        
        // 3. Set timesteps
        scheduler.setTimesteps(numInferenceSteps: numInferenceSteps, sigmas: sigmas, mu: mu)
        
        // 4. Prepare initial random noise latent [1, 4, 128, 256]
        var latents = generateRandomLatent(
            channels: latentChannels, height: latentHeight, width: latentWidth
        )
        
        // 5. Prepare time_ids [1, 2] = [width, height]
        let timeIds = try createMLMultiArray(shape: [1, 2], values: [Float(width), Float(height)])
        
        // 6. Prepare zero image latent for generate mode [1, 4, 128, 128]
        // let zeroImageLatent = try MLMultiArray(shape: [1, 4, 128, 128] as [NSNumber], dataType: .float16)
        let imageLatent: MLMultiArray
        if let conditionLatent {
            imageLatent = conditionLatent
        } else {
            let latH = NSNumber(value: height / 8)
            let latW = NSNumber(value: width / 8)
            imageLatent = try MLMultiArray(shape: [1, 4, latH, latW], dataType: .float16)
            memset(imageLatent.dataPointer, 0, imageLatent.count * 2)  // Float16 = 2 bytes
        }
        
        print("[Pipeline] Starting denoising loop (\(numInferenceSteps) steps)...")
        
        print("[Pipeline] Embedding loaded: hs shape=\(promptEmbeds.shape), mask shape=\(attentionMask.shape)")
        
        // Check embedding values
        let hsPtr = promptEmbeds.dataPointer.bindMemory(to: Float16.self, capacity: promptEmbeds.count)
        var hsMin: Float = Float.greatestFiniteMagnitude
        var hsMax: Float = -Float.greatestFiniteMagnitude
        for i in 0..<min(1000, promptEmbeds.count) {
            let v = Float(hsPtr[i])
            hsMin = min(hsMin, v)
            hsMax = max(hsMax, v)
        }
        print("[Pipeline] Embedding value range: [\(hsMin), \(hsMax)]")
        
        // 在 "Starting denoising loop" 之后加入
        print("[DEBUG] latents strides: \(latents.strides), shape: \(latents.shape)")
        print("[DEBUG] imageLatent strides: \(imageLatent.strides), shape: \(imageLatent.shape)")

        // 检查 latents 的值
        let latPtr = latents.dataPointer.bindMemory(to: Float16.self, capacity: latents.count)
        var latNaN = 0
        for i in 0..<latents.count { if Float(latPtr[i]).isNaN { latNaN += 1 } }
        print("[DEBUG] initial latents NaN count: \(latNaN)/\(latents.count)")
        
        // 7. Denoising loop
        printMemoryUsage("UNet循环开始前")
        var unetStepTimes: [Double] = []
        let unetLoopStart = CFAbsoluteTimeGetCurrent()
        
        for i in 0..<numInferenceSteps {
            let stepStart = CFAbsoluteTimeGetCurrent()
            
            let t = scheduler.currentTimestep()
            
            // Concatenate latents + zero_image in width dim → [1, 4, 128, 256]
            let modelInput = try concatenateWidth(latents: latents, condImage: imageLatent)
            // concatenateWidth 之后
            print("[DEBUG] modelInput strides: \(modelInput.strides), shape: \(modelInput.shape)")
            let miPtr = modelInput.dataPointer.bindMemory(to: Float16.self, capacity: modelInput.count)
            var miNaN = 0
            for i in 0..<modelInput.count { if Float(miPtr[i]).isNaN { miNaN += 1 } }
            print("[DEBUG] modelInput NaN count: \(miNaN)/\(modelInput.count)")
            
            // Prepare timestep MLMultiArray [1]
            let timestepArray = try createMLMultiArray(shape: [1], values: [Float(t)])
            
            // UNet forward
            let noisePred = try unet.predict(
                sample: modelInput,
                timestep: timestepArray,
                encoderHiddenStates: promptEmbeds,
                encoderAttentionMask: attentionMask,
                timeIds: timeIds
            )
            
            // Extract first half of width (noise_pred[..., :128])
            let noisePredCropped = try cropWidth(noisePred, targetWidth: latentWidth)
            
            // Debug: check noise pred range
//            let npPtr = noisePredCropped.dataPointer.bindMemory(to: Float16.self, capacity: noisePredCropped.count)
//            var npMin: Float = Float.greatestFiniteMagnitude
//            var npMax: Float = -Float.greatestFiniteMagnitude
//            var nanCount = 0
//            for j in 0..<noisePredCropped.count {
//                let v = Float(npPtr[j])
//                if v.isNaN { nanCount += 1; continue }
//                npMin = min(npMin, v)
//                npMax = max(npMax, v)
//            }
//            print("[Pipeline] Step \(i+1) noise_pred range: [\(npMin), \(npMax)], NaN count: \(nanCount)/\(noisePredCropped.count)")
            
            // Scheduler step
            let noisePredFlat = multiArrayToFloatArray(noisePredCropped)
            let latentsFlat = multiArrayToFloatArray(latents)
            let newLatentsFlat = scheduler.step(modelOutput: noisePredFlat, sample: latentsFlat)
            latents = try floatArrayToMLMultiArray(
                newLatentsFlat, shape: [1, NSNumber(value: latentChannels),
                                        NSNumber(value: latentHeight),
                                        NSNumber(value: latentWidth)]
            )
            
            let stepEnd = CFAbsoluteTimeGetCurrent()
            unetStepTimes.append(stepEnd - stepStart)
            
            progressHandler?(i + 1, numInferenceSteps)
        }
        let unetLoopEnd = CFAbsoluteTimeGetCurrent()
        // 打印 UNet 计时
        print("╔══════════════════════════════════════╗")
        print("║       UNet Step Breakdown            ║")
        print("╠══════════════════════════════════════╣")
        for (i, dt) in unetStepTimes.enumerated() {
            print(String(format: "║  Step %d:  %7.0f ms                 ║", i + 1, dt * 1000))
        }
        let avgStep = unetStepTimes.reduce(0, +) / Double(unetStepTimes.count) * 1000
        print(String(format: "║  Average: %7.0f ms/step            ║", avgStep))
        print(String(format: "║  Total:   %7.0f ms                 ║", (unetLoopEnd - unetLoopStart) * 1000))
        print("╚══════════════════════════════════════╝")
        
        // 8. VAE decode
        printMemoryUsage("VAE解码开始前")
        print("[Pipeline] Decoding latent to image...")
        // No scaling needed: scaling_factor=1, shift_factor=0 for TAESD-XL
        let vaeStart = CFAbsoluteTimeGetCurrent()
        let image = try vae.decode(latent: latents)
        printMemoryUsage("VAE解码完成")
        let vaeEnd = CFAbsoluteTimeGetCurrent()
        print("[VAE Decode] inference: \(Int((vaeEnd - vaeStart) * 1000)) ms")
        print("[Pipeline] Image generated! Shape: \(image.shape)")
        
//        let imgPtr = image.dataPointer.bindMemory(to: Float16.self, capacity: image.count)
//        var imgMin: Float = Float.greatestFiniteMagnitude
//        var imgMax: Float = -Float.greatestFiniteMagnitude
//        for i in 0..<min(1000, image.count) {
//            let v = Float(imgPtr[i])
//            imgMin = min(imgMin, v)
//            imgMax = max(imgMax, v)
//        }
//        print("[Pipeline] VAE output range: [\(imgMin), \(imgMax)]")
        
        return image
    }
    
    // MARK: - Helper Functions
    
    /// Generate random noise latent
    private func generateRandomLatent(channels: Int, height: Int, width: Int) -> MLMultiArray {
        let shape = [1, channels, height, width] as [NSNumber]
        let array = try! MLMultiArray(shape: shape, dataType: .float16)
        memset(array.dataPointer, 0, array.count * 2)  // 先清零，再填充随机值
        let count = array.count
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
        
        for i in stride(from: 0, to: count - 1, by: 2) {
            let u1 = Double.random(in: 0.0001...1.0)
            let u2 = Double.random(in: 0.0...1.0)
            let r = sqrt(-2.0 * log(u1))
            let theta = 2.0 * Double.pi * u2
            ptr[i] = Float16(Float(r * cos(theta)))
            if i + 1 < count {
                ptr[i + 1] = Float16(Float(r * sin(theta)))
            }
        }
        return array
    }
    
    /// Concatenate two [1,C,H,W] arrays along width dimension
    private func concatenateWidth(latents: MLMultiArray, condImage: MLMultiArray) throws -> MLMultiArray {        
        let c = latents.shape[1].intValue
        let h = latents.shape[2].intValue
        let w1 = latents.shape[3].intValue
        let w2 = condImage.shape[3].intValue
        let totalW = w1 + w2
        
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: c), NSNumber(value: h), NSNumber(value: totalW)],
            dataType: .float16
        )
        
        let dstPtr = result.dataPointer.bindMemory(to: Float16.self, capacity: c * h * totalW)
        let srcPtr1 = latents.dataPointer.bindMemory(to: Float16.self, capacity: c * h * w1)
        let srcPtr2 = condImage.dataPointer.bindMemory(to: Float16.self, capacity: c * h * w2)
        
        for ci in 0..<c {
            for hi in 0..<h {
                let srcOff1 = ci * h * w1 + hi * w1
                let srcOff2 = ci * h * w2 + hi * w2
                let dstOff  = ci * h * totalW + hi * totalW
                // 复制 latents 行
                memcpy(dstPtr + dstOff, srcPtr1 + srcOff1, w1 * MemoryLayout<Float16>.size)
                // 复制 condImage 行
                memcpy(dstPtr + dstOff + w1, srcPtr2 + srcOff2, w2 * MemoryLayout<Float16>.size)
            }
        }
        return result
    }
    
    /// Crop MLMultiArray width: [1,C,H,W] → [1,C,H,targetWidth]
    private func cropWidth(_ array: MLMultiArray, targetWidth: Int) throws -> MLMultiArray {
        let c = array.shape[1].intValue
        let h = array.shape[2].intValue
        let fullW = array.shape[3].intValue
        
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: c), NSNumber(value: h), NSNumber(value: targetWidth)],
            dataType: .float16
        )
        
        let srcPtr = array.dataPointer.bindMemory(to: Float16.self, capacity: c * h * fullW)
        let dstPtr = result.dataPointer.bindMemory(to: Float16.self, capacity: c * h * targetWidth)
        
        for ci in 0..<c {
            for hi in 0..<h {
                let srcOff = ci * h * fullW + hi * fullW
                let dstOff = ci * h * targetWidth + hi * targetWidth
                memcpy(dstPtr + dstOff, srcPtr + srcOff, targetWidth * MemoryLayout<Float16>.size)
            }
        }
        return result
    }
    
    /// Convert MLMultiArray to [Float]
    private func multiArrayToFloatArray(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            result[i] = Float(ptr[i])
        }
        return result
    }
    
    /// Convert [Float] to MLMultiArray
    private func floatArrayToMLMultiArray(_ array: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float16)
        let ptr = result.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        for i in 0..<array.count {
            ptr[i] = Float16(array[i])
        }
        return result
    }
    
    /// Create MLMultiArray from values
    private func createMLMultiArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let array = try MLMultiArray(shape: nsShape, dataType: .float16)
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: values.count)
        for (i, v) in values.enumerated() {
            ptr[i] = Float16(v)
        }
        return array
    }
}
