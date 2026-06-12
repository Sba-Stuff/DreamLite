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

import Accelerate
import CoreML
import Foundation

/// Main DreamLite inference pipeline.
/// Orchestrates: Text Encoder → UNet denoising loop → VAE decode → image.
///
/// The UNet operates in an "in-context" fashion: at every step we concatenate the
/// noisy latent with the conditioning image latent along the width axis, run the
/// UNet, then crop the first half of the prediction back out.
class DreamLitePipeline {

    let unet: UNetModel
    let vae: VAEDecoder
    let scheduler: FluxScheduler

    // Latent dimensions (1024×1024 → 128×128 with the 8× VAE downsample).
    let latentChannels = 4
    let latentHeight = 128
    let latentWidth = 128
    let inContextWidth = 256       // latentWidth × 2 (noise || cond_image concatenated)
    let hiddenSize = 2048           // Qwen3-VL output dim

    init() throws {
        print("[Pipeline] Loading models...")
        self.unet = try UNetModel()
        self.vae = try VAEDecoder()
        self.scheduler = FluxScheduler()
        print("[Pipeline] Ready ✓")
    }

    // MARK: - Generate Image

    /// Generate an image from text (and optionally an image) embeddings.
    /// - Parameters:
    ///   - promptEmbeds: [1, N, 2048] text embeddings from Qwen3-VL.
    ///   - attentionMask: [1, N] attention mask.
    ///   - numInferenceSteps: number of denoising steps.
    ///   - width / height: output image size in pixels.
    ///   - conditionLatent: optional [1, 4, H/8, W/8] condition latent for edit mode.
    ///   - progressHandler: callback `(currentStep, totalSteps)`.
    /// - Returns: [1, 3, H, W] decoded image as MLMultiArray.
    func generate(
        promptEmbeds: MLMultiArray,
        attentionMask: MLMultiArray,
        numInferenceSteps: Int = 4,
        width: Int = 1024,
        height: Int = 1024,
        conditionLatent: MLMultiArray? = nil,
        progressHandler: ((Int, Int) -> Void)? = nil
    ) throws -> MLMultiArray {
        // 1. Sigma schedule (matching Python: np.linspace(1.0, 1/N, N)).
        let total: Double = Double(numInferenceSteps)
        var sigmas: [Double] = []
        sigmas.reserveCapacity(numInferenceSteps)
        for i in 0..<numInferenceSteps {
            let step: Double = Double(i)
            let slope: Double = (1.0 - 1.0 / total) / (total - 1.0)
            let value: Double = 1.0 - step * slope
            sigmas.append(value)
        }

        // 2. Dynamic shift (mu) based on image sequence length.
        let imageSeqLen = (latentHeight * latentWidth) / 4
        let mu = FluxScheduler.calculateShift(
            imageSeqLen: imageSeqLen,
            baseSeqLen: scheduler.baseImageSeqLen,
            maxSeqLen: scheduler.maxImageSeqLen,
            baseShift: scheduler.baseShift,
            maxShift: scheduler.maxShift
        )

        // 3. Set scheduler timesteps.
        scheduler.setTimesteps(numInferenceSteps: numInferenceSteps, sigmas: sigmas, mu: mu)

        // 4. Initial random latent and conditioning latent.
        var latents = generateRandomLatent(
            channels: latentChannels, height: latentHeight, width: latentWidth
        )

        let timeIds = try createMLMultiArray(shape: [1, 2], values: [Float(width), Float(height)])

        let imageLatent: MLMultiArray
        if let conditionLatent {
            imageLatent = conditionLatent
        } else {
            let latH = NSNumber(value: height / 8)
            let latW = NSNumber(value: width / 8)
            imageLatent = try MLMultiArray(shape: [1, 4, latH, latW], dataType: .float16)
            memset(imageLatent.dataPointer, 0, imageLatent.count * 2)  // Float16 = 2 bytes
        }

        // 5. Denoising loop.
        print("[Pipeline] Denoising loop (\(numInferenceSteps) steps)...")
        var unetStepTimes: [Double] = []
        let unetLoopStart = CFAbsoluteTimeGetCurrent()

        for i in 0..<numInferenceSteps {
            let stepStart = CFAbsoluteTimeGetCurrent()

            let t = scheduler.currentTimestep()
            let modelInput = try concatenateWidth(latents: latents, condImage: imageLatent)
            let timestepArray = try createMLMultiArray(shape: [1], values: [Float(t)])

            let noisePred = try unet.predict(
                sample: modelInput,
                timestep: timestepArray,
                encoderHiddenStates: promptEmbeds,
                encoderAttentionMask: attentionMask,
                timeIds: timeIds
            )

            // Take only the noise side (first half of width).
            let noisePredCropped = try cropWidth(noisePred, targetWidth: latentWidth)

            // Scheduler step.
            let noisePredFlat = multiArrayToFloatArray(noisePredCropped)
            let latentsFlat = multiArrayToFloatArray(latents)
            let newLatentsFlat = scheduler.step(modelOutput: noisePredFlat, sample: latentsFlat)
            latents = try floatArrayToMLMultiArray(
                newLatentsFlat,
                shape: [1, NSNumber(value: latentChannels),
                        NSNumber(value: latentHeight),
                        NSNumber(value: latentWidth)]
            )

            unetStepTimes.append(CFAbsoluteTimeGetCurrent() - stepStart)
            progressHandler?(i + 1, numInferenceSteps)
        }
        let unetLoopEnd = CFAbsoluteTimeGetCurrent()

        // Print UNet timing.
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

        // 6. VAE decode (TAESD-XL: scaling_factor=1, shift_factor=0 → no rescale).
        let vaeStart = CFAbsoluteTimeGetCurrent()
        let image = try vae.decode(latent: latents)
        print("[VAE Decode] inference: \(Int((CFAbsoluteTimeGetCurrent() - vaeStart) * 1000)) ms")

        return image
    }

    // MARK: - Helpers

    /// Box-Muller normal noise into a Float16 MLMultiArray.
    private func generateRandomLatent(channels: Int, height: Int, width: Int) -> MLMultiArray {
        let shape = [1, channels, height, width] as [NSNumber]
        let array = try! MLMultiArray(shape: shape, dataType: .float16)
        memset(array.dataPointer, 0, array.count * 2)
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

    /// Concatenate two [1, C, H, W] arrays along the width axis.
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
                memcpy(dstPtr + dstOff,         srcPtr1 + srcOff1, w1 * MemoryLayout<Float16>.size)
                memcpy(dstPtr + dstOff + w1,    srcPtr2 + srcOff2, w2 * MemoryLayout<Float16>.size)
            }
        }
        return result
    }

    /// Crop along width: [1, C, H, W] → [1, C, H, targetWidth].
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

    private func multiArrayToFloatArray(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            result[i] = Float(ptr[i])
        }
        return result
    }

    private func floatArrayToMLMultiArray(_ array: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float16)
        let ptr = result.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        for i in 0..<array.count {
            ptr[i] = Float16(array[i])
        }
        return result
    }

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
