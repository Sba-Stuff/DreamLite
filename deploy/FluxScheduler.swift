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
import Accelerate

/// FlowMatch Euler Discrete Scheduler (matching Python FlowMatchEulerDiscreteScheduler)
class FluxScheduler {
    
    // Config values (from scheduler_config.json)
    let numTrainTimesteps: Int = 1000
    let shift: Double = 3.0
    let useDynamicShifting: Bool = true
    let baseShift: Double = 0.5
    let maxShift: Double = 1.15
    let baseImageSeqLen: Int = 256
    let maxImageSeqLen: Int = 4096
    
    // State
    var sigmas: [Double] = []
    var timesteps: [Double] = []
    var numInferenceSteps: Int = 0
    private var stepIndex: Int = 0
    
    init() {}
    
    // MARK: - Calculate dynamic shift mu
    
    /// Calculate shift factor based on image sequence length
    /// Matches Python: calculate_shift() in pipeline_utils.py
    static func calculateShift(
        imageSeqLen: Int,
        baseSeqLen: Int = 256,
        maxSeqLen: Int = 4096,
        baseShift: Double = 0.5,
        maxShift: Double = 1.16
    ) -> Double {
        let m = (maxShift - baseShift) / Double(maxSeqLen - baseSeqLen)
        let b = baseShift - m * Double(baseSeqLen)
        return Double(imageSeqLen) * m + b
    }
    
    // MARK: - Time shift (exponential)
    
    /// Apply exponential time shift: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
    /// Matches Python: _time_shift_exponential(mu, sigma, t)
    private func timeShiftExponential(mu: Double, sigma: Double, t: [Double]) -> [Double] {
        let expMu = exp(mu)
        return t.map { tVal in
            // Clamp t to avoid division by zero
            let tClamped = max(tVal, 1e-8)
            return expMu / (expMu + pow(1.0 / tClamped - 1.0, sigma))
        }
    }
    
    // MARK: - Set timesteps
    
    /// Set timesteps matching Python set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
    /// Pipeline calls: sigmas = np.linspace(1.0, 1/num_inference_steps, num_inference_steps)
    func setTimesteps(numInferenceSteps: Int, sigmas inputSigmas: [Double]? = nil, mu: Double) {
        self.numInferenceSteps = numInferenceSteps
        
        // 1. Prepare sigmas
        var currentSigmas: [Double]
        if let inputSigmas = inputSigmas {
            currentSigmas = inputSigmas
        } else {
            // Default: np.linspace(sigma_max_t, sigma_min_t, num_inference_steps) / num_train_timesteps
            currentSigmas = (0..<numInferenceSteps).map { i in
                let t = Double(numTrainTimesteps) - Double(i) * Double(numTrainTimesteps - 1) / Double(numInferenceSteps - 1)
                return t / Double(numTrainTimesteps)
            }
        }
        
        // 2. Apply dynamic shifting (exponential time shift)
        if useDynamicShifting {
            currentSigmas = timeShiftExponential(mu: mu, sigma: 1.0, t: currentSigmas)
        } else {
            currentSigmas = currentSigmas.map { s in
                shift * s / (1.0 + (shift - 1.0) * s)
            }
        }
        
        // 3. Compute timesteps = sigmas * num_train_timesteps
        self.timesteps = currentSigmas.map { $0 * Double(numTrainTimesteps) }
        
        // 4. Append terminal sigma (0.0) — matches: sigmas = cat([sigmas, zeros(1)])
        currentSigmas.append(0.0)
        self.sigmas = currentSigmas
        
        self.stepIndex = 0
    }
    
    // MARK: - Step
    
    /// Single Euler step: prev_sample = sample + (sigma_next - sigma) * model_output
    /// Matches Python step() method
    func step(
        modelOutput: [Float],
        sample: [Float]
    ) -> [Float] {
        let sigma = sigmas[stepIndex]
        let sigmaNext = sigmas[stepIndex + 1]
        let dt = Float(sigmaNext - sigma)
        
        // prev_sample = sample + dt * model_output  (using Accelerate for speed)
        var result = [Float](repeating: 0, count: sample.count)
        var dtVal = dt
        vDSP_vsma(modelOutput, 1, &dtVal, sample, 1, &result, 1, vDSP_Length(sample.count))
        
        // Advance step index
        stepIndex += 1
        
        return result
    }
    
    /// Get current timestep value for this step
    func currentTimestep() -> Double {
        return timesteps[stepIndex]
    }
}
