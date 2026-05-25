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

import SwiftUI
import PhotosUI
import CoreML
import MLX
import Accelerate

extension Color {
    static let dreamDeepBlue = Color(red: 0.08, green: 0.12, blue: 0.27)
    static let dreamMidBlue  = Color(red: 0.10, green: 0.22, blue: 0.42)
    static let dreamTeal     = Color(red: 0.12, green: 0.58, blue: 0.56)
    static let dreamCyan     = Color(red: 0.20, green: 0.78, blue: 0.72)
    static let dreamCard     = Color.white.opacity(0.08)
    static let dreamCardSolid = Color(red: 0.11, green: 0.15, blue: 0.28)
}

// MARK: - Resolution Presets

struct ResolutionPreset: Identifiable, Hashable {
    let id = UUID()
    let label: String
    let width: Int
    let height: Int

    var displayName: String {
        if width == height { return "\(width)²" }
        return "\(width)×\(height)"
    }

    var aspectLabel: String {
        let gcd = Self.gcd(width, height)
        return "\(width/gcd):\(height/gcd)"
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        b == 0 ? a : gcd(b, a % b)
    }
}

let resolutionPresets: [ResolutionPreset] = [
    .init(label: "Wide",        width: 1216, height: 832),
    .init(label: "Photo",       width: 1152, height: 896),
    .init(label: "Square",      width: 1024, height: 1024),
    .init(label: "Portrait",    width: 896,  height: 1152),
    .init(label: "Narrow",      width: 832,  height: 1216),
]

struct ContentView: View {

    @State private var prompt: String = ""
    @State private var generatedImage: UIImage?
    @State private var isGenerating = false
    @State private var statusMessage = "Ready"
    @State private var progressSteps: (Int, Int) = (0, 0)

    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var sourceImage: UIImage?

    // Parameters
    @State private var inferenceSteps: Double = 4
    @State private var selectedResolution: ResolutionPreset = resolutionPresets[2] // 1024×1024
    @State private var showAdvanced = false

    @State private var showSavedToast = false

    @State private var pipeline: DreamLitePipeline?
    @State private var textEncoder: MLXTextEncoder?
    @State private var vaeEncoder: MLModel?

    // [优化4] FocusState 控制键盘收起
    @FocusState private var isPromptFocused: Bool

    private var isEditMode: Bool { sourceImage != nil }
    private var canRun: Bool { !isGenerating && !prompt.isEmpty }

    // [优化1] 进度比例 0~1
    private var stepProgress: Double {
        guard progressSteps.1 > 0 else { return 0 }
        return Double(progressSteps.0) / Double(progressSteps.1)
    }

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [.dreamDeepBlue, .dreamMidBlue, .dreamDeepBlue],
                startPoint: .top, endPoint: .bottom
            )
            .ignoresSafeArea()

            ScrollView(showsIndicators: false) {
                VStack(spacing: 0) {

                    // ── Header ──
                    HStack(alignment: .center) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("DreamLite")
                                .font(.system(size: 26, weight: .bold, design: .rounded))
                                .foregroundColor(.white)
                            Text(isEditMode ? "Edit Mode" : "Generate Mode")
                                .font(.caption.weight(.medium))
                                .foregroundColor(isEditMode ? .dreamCyan : .dreamTeal)
                        }
                        Spacer()
                        Image(systemName: isEditMode ? "pencil.and.outline" : "wand.and.stars")
                            .font(.title3)
                            .foregroundColor(.dreamCyan)
                            .frame(width: 40, height: 40)
                            .background(Color.dreamCard)
                            .clipShape(Circle())
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                    .padding(.bottom, 16)

                    // ── [优化2] Generating placeholder OR [优化3] Result with X button ──
                    if isGenerating {
                        // 生成中占位框
                        generatingPlaceholder
                            .padding(.horizontal, 20)
                            .padding(.bottom, 16)
                    } else if let generatedImage {
                        // [优化3] 生成结果 + 右上角 X
                        resultImageView(generatedImage)
                            .padding(.horizontal, 20)
                            .padding(.bottom, 16)
                            .transition(.opacity.combined(with: .scale(scale: 0.96)))
                    }

                    // ── Prompt Card ──
                    cardView {
                        VStack(alignment: .leading, spacing: 8) {
                            sectionLabel("Prompt")
                            TextField(
                                isEditMode ? "Describe the edit..." : "Describe the image...",
                                text: $prompt,
                                axis: .vertical
                            )
                            .lineLimit(3...6)
                            .font(.body)
                            .foregroundColor(.white)
                            .tint(.dreamCyan)
                            .focused($isPromptFocused) // [优化4]
                        }
                    }

                    // ── Reference Image Card ──
                    cardView {
                        PhotosPicker(
                            selection: $selectedPhotoItem,
                            matching: .images
                        ) {
                            if let sourceImage {
                                ZStack(alignment: .topTrailing) {
                                    Image(uiImage: sourceImage)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(maxHeight: 180)
                                        .cornerRadius(10)

                                    Button {
                                        withAnimation(.easeOut(duration: 0.2)) {
                                            self.sourceImage = nil
                                            self.selectedPhotoItem = nil
                                        }
                                    } label: {
                                        Image(systemName: "xmark.circle.fill")
                                            .font(.title3)
                                            .symbolRenderingMode(.palette)
                                            .foregroundStyle(.white, .black.opacity(0.5))
                                    }
                                    .padding(6)
                                }
                            } else {
                                HStack(spacing: 10) {
                                    Image(systemName: "photo.badge.plus")
                                        .font(.title3)
                                        .foregroundColor(.dreamTeal)
                                    Text("Add reference image for editing")
                                        .font(.subheadline)
                                        .foregroundColor(.white.opacity(0.6))
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.caption)
                                        .foregroundColor(.white.opacity(0.2))
                                }
                                .padding(.vertical, 14)
                            }
                        }
                        .onChange(of: selectedPhotoItem) { _, newItem in
                            Task {
                                if let data = try? await newItem?.loadTransferable(type: Data.self),
                                   let uiImage = UIImage(data: data) {
                                    withAnimation { sourceImage = uiImage }
                                }
                            }
                        }
                    }

                    // ── Advanced Settings (collapsible) ──
                    cardView {
                        DisclosureGroup(isExpanded: $showAdvanced) {
                            VStack(spacing: 16) {
                                Divider().background(Color.white.opacity(0.1))
                                    .padding(.top, 4)

                                // Resolution
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        sectionLabel("Resolution")
                                        Spacer()
                                        Text("\(selectedResolution.width)×\(selectedResolution.height)")
                                            .font(.system(.caption, design: .monospaced).weight(.bold))
                                            .foregroundColor(.dreamCyan)
                                    }

                                    LazyVGrid(
                                        columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 3),
                                        spacing: 8
                                    ) {
                                        ForEach(resolutionPresets) { preset in
                                            Button {
                                                selectedResolution = preset
                                            } label: {
                                                VStack(spacing: 4) {
                                                    let maxSide: CGFloat = 32
                                                    let aspect = CGFloat(preset.width) / CGFloat(preset.height)
                                                    let previewW = aspect >= 1 ? maxSide : maxSide * aspect
                                                    let previewH = aspect <= 1 ? maxSide : maxSide / aspect

                                                    RoundedRectangle(cornerRadius: 3)
                                                        .fill(selectedResolution.id == preset.id
                                                              ? Color.dreamCyan
                                                              : Color.white.opacity(0.15))
                                                        .frame(width: previewW, height: previewH)

                                                    Text(preset.label)
                                                        .font(.system(size: 10, weight: .medium))
                                                        .foregroundColor(selectedResolution.id == preset.id
                                                                         ? .dreamCyan : .white.opacity(0.5))

                                                    Text(preset.aspectLabel)
                                                        .font(.system(size: 9, design: .monospaced))
                                                        .foregroundColor(.white.opacity(0.3))
                                                }
                                                .frame(maxWidth: .infinity)
                                                .padding(.vertical, 8)
                                                .background(
                                                    RoundedRectangle(cornerRadius: 8)
                                                        .fill(selectedResolution.id == preset.id
                                                              ? Color.dreamCyan.opacity(0.12)
                                                              : Color.white.opacity(0.04))
                                                )
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 8)
                                                        .stroke(selectedResolution.id == preset.id
                                                                ? Color.dreamCyan.opacity(0.4)
                                                                : Color.clear, lineWidth: 1)
                                                )
                                            }
                                        }
                                    }
                                }

                                // Steps
                                HStack {
                                    sectionLabel("Steps")
                                    Spacer()
                                    Text("\(Int(inferenceSteps))")
                                        .font(.system(.subheadline, design: .rounded).weight(.bold))
                                        .foregroundColor(.dreamCyan)
                                        .frame(minWidth: 28)
                                }
                                Slider(value: $inferenceSteps, in: 4...20, step: 1)
                                    .tint(.dreamTeal)
                            }
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "slider.horizontal.3")
                                    .foregroundColor(.dreamTeal)
                                Text("Advanced Settings")
                                    .font(.subheadline.weight(.medium))
                                    .foregroundColor(.white.opacity(0.8))
                                Spacer()
                                if !showAdvanced {
                                    Text("\(selectedResolution.displayName) · \(Int(inferenceSteps))steps")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundColor(.white.opacity(0.35))
                                }
                            }
                        }
                        .tint(.dreamTeal)
                    }

                    // ── Run Button ──
                    Button {
                        Task { await runPipelineWithProfiling() }
                    } label: {
                        HStack(spacing: 8) {
                            if isGenerating {
                                ProgressView().tint(.white)
                            } else {
                                Image(systemName: isEditMode ? "pencil.and.outline" : "sparkles")
                            }
                            Text(isGenerating
                                 ? "Step \(progressSteps.0) / \(progressSteps.1)"
                                 : (isEditMode ? "Edit Image" : "Generate Image"))
                                .fontWeight(.semibold)
                        }
                        .font(.body)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 15)
                        .background(
                            canRun
                            ? AnyShapeStyle(LinearGradient(
                                colors: [.dreamTeal, .dreamCyan],
                                startPoint: .leading, endPoint: .trailing))
                            : AnyShapeStyle(Color.white.opacity(0.08))
                        )
                        .foregroundColor(canRun ? .white : .white.opacity(0.3))
                        .cornerRadius(14)
                    }
                    .disabled(!canRun)
                    .padding(.horizontal, 20)
                    .padding(.top, 4)
                    .padding(.bottom, 8)

                    // Status
                    if isGenerating || (statusMessage != "Ready" && generatedImage == nil) {
                        Text(statusMessage)
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.5))
                            .padding(.bottom, 4)
                    }
                }
                .padding(.bottom, 50)
            }
            .scrollDismissesKeyboard(.interactively) // [优化4] 滚动时渐进收起键盘

            // Saved toast
            if showSavedToast {
                VStack {
                    Spacer()
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.dreamCyan)
                        Text("Saved to Photos")
                            .font(.subheadline.weight(.medium))
                            .foregroundColor(.white)
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial)
                    .environment(\.colorScheme, .dark)
                    .cornerRadius(24)
                    .shadow(color: .black.opacity(0.2), radius: 8, y: 4)
                    .padding(.bottom, 50)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .preferredColorScheme(.dark)
        .animation(.easeInOut(duration: 0.25), value: isEditMode)
        .animation(.easeInOut(duration: 0.3), value: generatedImage != nil)
        .animation(.easeInOut(duration: 0.3), value: isGenerating)
        .animation(.spring(duration: 0.4), value: showSavedToast)
        // [优化4] 点击空白区域收起键盘
        .onTapGesture { isPromptFocused = false }
        .task { await loadModels() }
    }

    // MARK: - [优化2] Generating Placeholder

    private var generatingPlaceholder: some View {
        VStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.dreamCardSolid)
                .frame(height: 240)
                .overlay(
                    VStack(spacing: 16) {
                        // Pulsing icon
                        Image(systemName: isEditMode ? "pencil.and.outline" : "sparkles")
                            .font(.system(size: 32))
                            .foregroundColor(.dreamCyan.opacity(0.6))
                            .symbolEffect(.pulse, isActive: isGenerating)

                        Text("Step \(progressSteps.0) of \(progressSteps.1)")
                            .font(.subheadline.weight(.medium))
                            .foregroundColor(.white.opacity(0.7))

                        // [优化1] Linear progress bar
                        ProgressView(value: stepProgress)
                            .progressViewStyle(LinearProgressViewStyle(tint: .dreamCyan))
                            .frame(width: 180)

                        Text(statusMessage)
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.4))
                    }
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [.dreamTeal.opacity(0.4), .dreamCyan.opacity(0.4)],
                                startPoint: .topLeading, endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
        }
    }

    // MARK: - [优化3] Result Image with X Button

    private func resultImageView(_ image: UIImage) -> some View {
        ZStack(alignment: .topTrailing) {
            VStack(spacing: 10) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .cornerRadius(14)
                    .shadow(color: .black.opacity(0.3), radius: 12, y: 6)

                Button {
                    saveImageToAlbum(image)
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "square.and.arrow.down")
                        Text("Save to Photos")
                    }
                    .font(.subheadline.weight(.medium))
                    .foregroundColor(.dreamCyan)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 18)
                    .background(Capsule().fill(Color.dreamCyan.opacity(0.15)))
                }
            }

            // [优化3] X dismiss button
            Button {
                withAnimation(.easeOut(duration: 0.2)) {
                    generatedImage = nil
                }
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(.white.opacity(0.9), .black.opacity(0.55))
            }
            .padding(8)
        }
    }

    // MARK: - UI Components

    @ViewBuilder
    private func cardView<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.dreamCardSolid)
            )
            .padding(.horizontal, 20)
            .padding(.bottom, 12)
    }

    @ViewBuilder
    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .foregroundColor(.dreamTeal)
            .textCase(.uppercase)
            .tracking(0.5)
    }

    private func saveImageToAlbum(_ image: UIImage) {
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        showSavedToast = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            showSavedToast = false
        }
    }

    // MARK: - Load models

    private func loadModels() async {
        printMemoryUsage("App启动基础内存")
        statusMessage = "Loading models..."
        do {
            guard let enc = MLXTextEncoder() else {
                statusMessage = "Failed to create text encoder"
                return
            }
            try await enc.loadModel()
            printMemoryUsage("Text Encoder加载完成")
            textEncoder = enc
            let pipe = try DreamLitePipeline()
            pipeline = pipe

            if let encoderURL = Bundle.main.url(forResource: "vae_encoder", withExtension: "mlmodelc") {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndNeuralEngine
                vaeEncoder = try MLModel(contentsOf: encoderURL, configuration: config)
            }
            statusMessage = "Ready"
        } catch {
            statusMessage = "Load failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Performance Profiling
    
    /// 在 runPipeline 中替换或并行调用，打印各阶段耗时
    private func runPipelineWithProfiling() async {
        guard let textEncoder, let pipeline else { return }
        isGenerating = true
        isPromptFocused = false
        generatedImage = nil
        progressSteps = (0, Int(inferenceSteps))
        
        let steps = Int(inferenceSteps)
        let w = selectedResolution.width
        let h = selectedResolution.height
        let originalSourceSize = sourceImage?.size
        
        var timings: [(String, Double)] = []
        let totalStart = CFAbsoluteTimeGetCurrent()
        
        do {
            let embeds: MLMultiArray
            let mask: MLMultiArray
            var conditionLatent: MLMultiArray? = nil
            
            // ── Stage 1: Text Encoder (MLX 4-bit) ──
            let t1 = CFAbsoluteTimeGetCurrent()
            
            if let sourceImage {
                guard let vaeEncoder else {
                    statusMessage = "VAE encoder not loaded"
                    isGenerating = false
                    return
                }
                let editPrefix = "[Edit]: A diptych with two side-by-side images of the same scene. Compared to the right side, the left one has "
                let fullEditPrompt = editPrefix + prompt
                
                statusMessage = "Encoding edit prompt..."
                let encoded = try textEncoder.encodeEditPrompt(
                    instruction: fullEditPrompt,
                    sourceImage: sourceImage
                )
                embeds = encoded.hiddenStates
                mask = encoded.attentionMask
                
                let t1End = CFAbsoluteTimeGetCurrent()
                timings.append(("Text Encoder (edit + vision)", t1End - t1))
                
                // ── Stage 2: VAE Encoder (CoreML) ──
                let t2 = CFAbsoluteTimeGetCurrent()
                statusMessage = "Encoding source image..."
                conditionLatent = try encodeSourceImageToLatent(
                    sourceImage, vaeEncoder: vaeEncoder,
                    width: 1024, height: 1024
                )
                let t2End = CFAbsoluteTimeGetCurrent()
                timings.append(("VAE Encoder", t2End - t2))
                
            } else {
                statusMessage = "Encoding prompt..."
                let encoded = try textEncoder.encodePrompt(prompt: "[Generate]: \(prompt)")
                embeds = encoded.hiddenStates
                mask = encoded.attentionMask
                
                let t1End = CFAbsoluteTimeGetCurrent()
                timings.append(("Text Encoder (generate)", t1End - t1))
            }
            
            // ── Stage 3: UNet Diffusion (CoreML) ──
            let t3 = CFAbsoluteTimeGetCurrent()
            statusMessage = "Generating..."
            
            let latent = try await Task.detached(priority: .userInitiated) {
                try pipeline.generate(
                    promptEmbeds: embeds,
                    attentionMask: mask,
                    numInferenceSteps: steps,
                    width: w,
                    height: h,
                    conditionLatent: conditionLatent,
                    progressHandler: { step, total in
                        DispatchQueue.main.async {
                            self.progressSteps = (step, total)
                        }
                    }
                )
            }.value
            
            let t3End = CFAbsoluteTimeGetCurrent()
            timings.append(("UNet Total (\(steps) steps) + VAE Decoder", t3End - t3))
            
            // ── Stage 4: VAE Decoder (CoreML) ──
            let t4 = CFAbsoluteTimeGetCurrent()
            statusMessage = "Postprocessing..."
            
            var uiImage: UIImage? = nil
            if var decoded = mlArrayToUIImage(latent) {
                if isEditMode, let origSize = originalSourceSize {
                    let origW = Int(origSize.width)
                    let origH = Int(origSize.height)
                    if origW != 1024 || origH != 1024 {
                        decoded = resizeUIImage(decoded, to: CGSize(width: origW, height: origH))
                    }
                }
                uiImage = decoded
            }
            
            let t4End = CFAbsoluteTimeGetCurrent()
            timings.append(("Postprocess (MLArray->UIImage)", t4End - t4))
            
            let totalEnd = CFAbsoluteTimeGetCurrent()
            
            // ── Print Profiling Report ──
            print("╔══════════════════════════════════════════════════╗")
            print("║          DreamLite Performance Profile           ║")
            print("╠══════════════════════════════════════════════════╣")
            for (name, duration) in timings {
                let ms = duration * 1000
                let bar = String(repeating: "#", count: min(Int(ms / 50), 30))
                print(String(format: "║ %-37s %7.0f ms ║", (name as NSString).utf8String!, ms))
                print(String(format: "║ %-48s ║", (bar as NSString).utf8String!))
            }
            print("╠══════════════════════════════════════════════════╣")
            print(String(format: "║ TOTAL                                 %7.0f ms ║", (totalEnd - totalStart) * 1000))
            print("╚══════════════════════════════════════════════════╝")

            
            // Update UI
            if let uiImage {
                generatedImage = uiImage
                statusMessage = String(format: "Done in %.1fs", totalEnd - totalStart)
            } else {
                statusMessage = "Decode failed"
            }
            
        } catch {
            statusMessage = "Error: \(error.localizedDescription)"
        }
        isGenerating = false
    }

    // MARK: - VAE encode (Accelerate 优化版)

    private func encodeSourceImageToLatent(
        _ image: UIImage, vaeEncoder: MLModel, width: Int, height: Int
    ) throws -> MLMultiArray {
        let resized = resizeUIImage(image, to: CGSize(width: width, height: height))

        guard let cgImage = resized.cgImage else {
            throw NSError(domain: "DreamLite", code: -3, userInfo: [NSLocalizedDescriptionKey: "CGImage failed"])
        }
        let w = cgImage.width, h = cgImage.height
        let channelSize = w * h
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawPixels = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &rawPixels, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(domain: "DreamLite", code: -4, userInfo: [NSLocalizedDescriptionKey: "CGContext failed"])
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        // ── 1. RGBA interleaved → Planar R, G, B (UInt8) ──
        var rU8 = [UInt8](repeating: 0, count: channelSize)
        var gU8 = [UInt8](repeating: 0, count: channelSize)
        var bU8 = [UInt8](repeating: 0, count: channelSize)
        var aU8 = [UInt8](repeating: 0, count: channelSize)

        rawPixels.withUnsafeMutableBufferPointer { srcBuf in
            rU8.withUnsafeMutableBufferPointer { rBuf in
                gU8.withUnsafeMutableBufferPointer { gBuf in
                    bU8.withUnsafeMutableBufferPointer { bBuf in
                        aU8.withUnsafeMutableBufferPointer { aBuf in
                            var src = vImage_Buffer(data: srcBuf.baseAddress!,
                                                    height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
                            var rDst = vImage_Buffer(data: rBuf.baseAddress!,
                                                     height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
                            var gDst = vImage_Buffer(data: gBuf.baseAddress!,
                                                     height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
                            var bDst = vImage_Buffer(data: bBuf.baseAddress!,
                                                     height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
                            var aDst = vImage_Buffer(data: aBuf.baseAddress!,
                                                     height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
                            // RGBA → R, G, B, A planar
                            vImageConvert_ARGB8888toPlanar8(&src, &rDst, &gDst, &bDst, &aDst, 0)
                        }
                    }
                }
            }
        }

        // ── 2. UInt8 → Float32 ──
        var rFloat = [Float](repeating: 0, count: channelSize)
        var gFloat = [Float](repeating: 0, count: channelSize)
        var bFloat = [Float](repeating: 0, count: channelSize)
        let n = vDSP_Length(channelSize)

        vDSP_vfltu8(&rU8, 1, &rFloat, 1, n)
        vDSP_vfltu8(&gU8, 1, &gFloat, 1, n)
        vDSP_vfltu8(&bU8, 1, &bFloat, 1, n)

        // ── 3. [0, 255] → [-1, 1]:  val / 127.5 - 1.0 ──
        var divVal: Float = 127.5
        var subVal: Float = -1.0

        func normalizeChannel(_ channel: inout [Float]) {
            channel.withUnsafeMutableBufferPointer { buf in
                let ptr = buf.baseAddress!
                vDSP_vsdiv(ptr, 1, &divVal, ptr, 1, n)   // /127.5
                vDSP_vsadd(ptr, 1, &subVal, ptr, 1, n)   // -1.0
            }
        }
        normalizeChannel(&rFloat)
        normalizeChannel(&gFloat)
        normalizeChannel(&bFloat)

        // ── 4. Float32 → Float16, 写入 MLMultiArray CHW layout ──
        let mlInput = try MLMultiArray(
            shape: [1, 3, NSNumber(value: h), NSNumber(value: w)], dataType: .float16)
        let dstPtr = mlInput.dataPointer.bindMemory(to: UInt16.self, capacity: channelSize * 3)

        rFloat.withUnsafeBufferPointer { rBuf in
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: rBuf.baseAddress!),
                                    height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 4)
            var dst = vImage_Buffer(data: UnsafeMutableRawPointer(dstPtr),
                                    height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 2)
            vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
        }
        gFloat.withUnsafeBufferPointer { gBuf in
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: gBuf.baseAddress!),
                                    height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 4)
            var dst = vImage_Buffer(data: UnsafeMutableRawPointer(dstPtr.advanced(by: channelSize)),
                                    height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 2)
            vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
        }
        bFloat.withUnsafeBufferPointer { bBuf in
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: bBuf.baseAddress!),
                                    height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 4)
            var dst = vImage_Buffer(data: UnsafeMutableRawPointer(dstPtr.advanced(by: channelSize * 2)),
                                    height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 2)
            vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
        }

        // ── 5. VAE encoder 推理 ──
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(multiArray: mlInput)])
        let output = try vaeEncoder.prediction(from: input)
        guard let latent = output.featureValue(for: "latent")?.multiArrayValue else {
            throw NSError(domain: "DreamLite", code: -6, userInfo: [NSLocalizedDescriptionKey: "VAE output failed"])
        }
        return latent
    }

    // MARK: - [优化5] Resize helper

    private func resizeUIImage(_ image: UIImage, to targetSize: CGSize) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0  // ★ 关键！强制像素=逻辑尺寸，不受 Retina 影响
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    // MARK: - MLMultiArray → UIImage

//    private func mlArrayToUIImage(_ array: MLMultiArray) -> UIImage? {
//        let shape = array.shape.map { $0.intValue }
//        guard shape.count == 4, shape[1] == 3 else { return nil }
//        let h = shape[2], w = shape[3]
//        let total = shape.reduce(1, *)
//        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: total)
//        var pixels = [UInt8](repeating: 255, count: w * h * 4)
//        for y in 0..<h {
//            for x in 0..<w {
//                for c in 0..<3 {
//                    let val = Float(ptr[c * h * w + y * w + x])
//                    let clamped = min(max((val + 1.0) / 2.0, 0), 1)
//                    pixels[(y * w + x) * 4 + c] = UInt8(clamped * 255)
//                }
//            }
//        }
//        let colorSpace = CGColorSpaceCreateDeviceRGB()
//        guard let imgCtx = CGContext(
//            data: &pixels, width: w, height: h,
//            bitsPerComponent: 8, bytesPerRow: w * 4,
//            space: colorSpace,
//            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
//        ), let cgImage = imgCtx.makeImage() else { return nil }
//        return UIImage(cgImage: cgImage)
//    }
    // MARK: - MLMultiArray → UIImage (Accelerate 优化版 ~10ms vs 原始 ~300ms)

    private func mlArrayToUIImage(_ array: MLMultiArray) -> UIImage? {
        let shape = array.shape.map { $0.intValue }
        guard shape.count == 4, shape[1] == 3 else { return nil }
        let h = shape[2], w = shape[3]
        let channelSize = h * w

        let srcPtr = array.dataPointer.bindMemory(to: UInt16.self, capacity: channelSize * 3)

        // ── 1. Float16 → Float32 (3通道批量) ──
        var rFloat = [Float](repeating: 0, count: channelSize)
        var gFloat = [Float](repeating: 0, count: channelSize)
        var bFloat = [Float](repeating: 0, count: channelSize)

        var rSrc16 = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr),
                                   height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 2)
        var gSrc16 = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.advanced(by: channelSize)),
                                   height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 2)
        var bSrc16 = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.advanced(by: channelSize * 2)),
                                   height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 2)

        rFloat.withUnsafeMutableBufferPointer { buf in
            var dst = vImage_Buffer(data: buf.baseAddress!, height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 4)
            vImageConvert_Planar16FtoPlanarF(&rSrc16, &dst, 0)
        }
        gFloat.withUnsafeMutableBufferPointer { buf in
            var dst = vImage_Buffer(data: buf.baseAddress!, height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 4)
            vImageConvert_Planar16FtoPlanarF(&gSrc16, &dst, 0)
        }
        bFloat.withUnsafeMutableBufferPointer { buf in
            var dst = vImage_Buffer(data: buf.baseAddress!, height: 1, width: vImagePixelCount(channelSize), rowBytes: channelSize * 4)
            vImageConvert_Planar16FtoPlanarF(&bSrc16, &dst, 0)
        }

        // ── 2. (val + 1.0) * 127.5, clamp [0, 255] ──
        var addVal: Float = 1.0
        var mulVal: Float = 127.5
        var lo: Float = 0.0
        var hi: Float = 255.0
        let n = vDSP_Length(channelSize)

        func processChannel(_ channel: inout [Float]) {
                channel.withUnsafeMutableBufferPointer { buf in
                    let ptr = buf.baseAddress!
                    vDSP_vsadd(ptr, 1, &addVal, ptr, 1, n)
                    vDSP_vsmul(ptr, 1, &mulVal, ptr, 1, n)
                    vDSP_vclip(ptr, 1, &lo, &hi, ptr, 1, n)
                }
            }
        
        processChannel(&rFloat)
        processChannel(&gFloat)
        processChannel(&bFloat)
        

        // ── 3. Float32 → UInt8 ──
        var rU8 = [UInt8](repeating: 0, count: channelSize)
        var gU8 = [UInt8](repeating: 0, count: channelSize)
        var bU8 = [UInt8](repeating: 0, count: channelSize)

        vDSP_vfixu8(&rFloat, 1, &rU8, 1, n)
        vDSP_vfixu8(&gFloat, 1, &gU8, 1, n)
        vDSP_vfixu8(&bFloat, 1, &bU8, 1, n)

        // ── 4. Planar R,G,B → Interleaved ARGB ──
        var pixels = [UInt8](repeating: 0, count: channelSize * 4)
        var alphaPlane = [UInt8](repeating: 255, count: channelSize)

        alphaPlane.withUnsafeMutableBufferPointer { aBuf in
            rU8.withUnsafeMutableBufferPointer { rBuf in
                gU8.withUnsafeMutableBufferPointer { gBuf in
                    bU8.withUnsafeMutableBufferPointer { bBuf in
                        pixels.withUnsafeMutableBufferPointer { pxBuf in
                            var aSrc = vImage_Buffer(data: aBuf.baseAddress!, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
                            var rSrc = vImage_Buffer(data: rBuf.baseAddress!, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
                            var gSrc = vImage_Buffer(data: gBuf.baseAddress!, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
                            var bSrc = vImage_Buffer(data: bBuf.baseAddress!, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
                            var dst  = vImage_Buffer(data: pxBuf.baseAddress!, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)

                            // 参数顺序: A, R, G, B → 输出 ARGB 交错格式
                            vImageConvert_Planar8toARGB8888(&aSrc, &rSrc, &gSrc, &bSrc, &dst, 0)
                        }
                    }
                }
            }
        }

        // ── 5. CGImage (ARGB) → UIImage ──
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixels,
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue  // ARGB
        ), let cgImage = ctx.makeImage() else { return nil }

        return UIImage(cgImage: cgImage)
    }
}
