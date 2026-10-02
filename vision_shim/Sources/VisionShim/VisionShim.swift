// VisionShim — a thin Swift → C ABI bridge over Apple Vision.
//
// Compiled to a dylib (`swiftc -O -emit-library`) and dlopen'd from Rust.
// All functions are synchronous (VNImageRequestHandler.perform is blocking)
// and return normalized geometry, so no Rust-side Vision knowledge is needed.
//
// Coordinate convention: normalization uses top-left origin (Vision uses
// bottom-left); we convert. Boxes/points are normalized to [0,1] of the frame.

import Foundation
import Vision
import CoreVideo
import ImageIO

private let SALIENCY_GRID: Int = 48

// MARK: - version

@_cdecl("shim_version")
public func shim_version() -> Int32 { 1 }

// MARK: - face rectangles

/// Detect faces. Fills `out` with `count` records of 5 floats:
///     [x, y, w, h, confidence]  (normalized, top-left origin)
/// Returns the number of faces written (≤ max_faces), or -1 on error.
@_cdecl("shim_faces")
public func shim_faces(path: UnsafePointer<CChar>?, out: UnsafeMutablePointer<Float>?, max_faces: Int32) -> Int32 {
    guard let cg = loadCGImage(path) else { return -1 }
    guard let out = out, max_faces > 0 else { return 0 }
    let req = VNDetectFaceRectanglesRequest()
    req.usesCPUOnly = false
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try handler.perform([req])
    } catch {
        return -1
    }
    var count = 0
    for obs in (req.results ?? []) {
        if count >= Int(max_faces) { break }
        let b = obs.boundingBox // bottom-left origin
        out[count * 5 + 0] = Float(b.origin.x)
        out[count * 5 + 1] = Float(1.0 - b.origin.y - b.size.height)
        out[count * 5 + 2] = Float(b.size.width)
        out[count * 5 + 3] = Float(b.size.height)
        out[count * 5 + 4] = Float(obs.confidence)
        count += 1
    }
    return Int32(count)
}

// MARK: - saliency heatmaps

/// Attention-based saliency. Fills `out` (SALIENCY_GRID × SALIENCY_GRID floats,
/// 0..1, row-major, top-left) and sets `out_w`/`out_h`.
/// Returns grid cell count, or -1 on error.
@_cdecl("shim_saliency_attention")
public func shim_saliency_attention(path: UnsafePointer<CChar>?, out: UnsafeMutablePointer<Float>?, out_w: UnsafeMutablePointer<Int32>?, out_h: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let cg = loadCGImage(path) else { return -1 }
    guard let out = out, let out_w = out_w, let out_h = out_h else { return 0 }
    let grid = saliencyGrid(makeRequest: VNGenerateAttentionBasedSaliencyImageRequest(), image: cg)
    guard let grid else { return -1 }
    copyGrid(grid, out: out, out_w: out_w, out_h: out_h)
    return Int32(SALIENCY_GRID * SALIENCY_GRID)
}

/// Objectness-based saliency. Same contract as attention variant.
@_cdecl("shim_saliency_objectness")
public func shim_saliency_objectness(path: UnsafePointer<CChar>?, out: UnsafeMutablePointer<Float>?, out_w: UnsafeMutablePointer<Int32>?, out_h: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let cg = loadCGImage(path) else { return -1 }
    guard let out = out, let out_w = out_w, let out_h = out_h else { return 0 }
    let grid = saliencyGrid(makeRequest: VNGenerateObjectnessBasedSaliencyImageRequest(), image: cg)
    guard let grid else { return -1 }
    copyGrid(grid, out: out, out_w: out_w, out_h: out_h)
    return Int32(SALIENCY_GRID * SALIENCY_GRID)
}

private func saliencyGrid(makeRequest: VNImageBasedRequest, image: CGImage) -> [Float]? {
    let req = makeRequest
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([req])
    } catch {
        return nil
    }
    guard let obs = req.results?.first as? VNSaliencyImageObservation
    else { return nil }
    return extractGrid(obs.pixelBuffer)
}

private func extractGrid(_ pb: CVPixelBuffer) -> [Float]? {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
    let w = CVPixelBufferGetWidth(pb)
    let h = CVPixelBufferGetHeight(pb)
    let bpr = CVPixelBufferGetBytesPerRow(pb)

    var grid = [Float](repeating: 0, count: SALIENCY_GRID * SALIENCY_GRID)
    var cells = [Int](repeating: 0, count: SALIENCY_GRID * SALIENCY_GRID)
    let ptr = base.bindMemory(to: UInt8.self, capacity: bpr * h)
    for y in 0..<h {
        let row = ptr + y * bpr
        for x in 0..<w {
            let gy = min(SALIENCY_GRID - 1, y * SALIENCY_GRID / h)
            let gx = min(SALIENCY_GRID - 1, x * SALIENCY_GRID / w)
            let g = gy * SALIENCY_GRID + gx
            grid[g] += Float(row[x]) / 255.0
            cells[g] += 1
        }
    }
    for i in 0..<grid.count where cells[i] > 0 {
        grid[i] /= Float(cells[i])
    }
    return grid
}

private func copyGrid(_ grid: [Float], out: UnsafeMutablePointer<Float>, out_w: UnsafeMutablePointer<Int32>, out_h: UnsafeMutablePointer<Int32>) {
    for i in 0..<grid.count { out[i] = grid[i] }
    out_w.pointee = Int32(SALIENCY_GRID)
    out_h.pointee = Int32(SALIENCY_GRID)
}

// MARK: - face landmarks (mouth region) for active-speaker resolution

/// Detect faces + mouth landmarks. Out layout per face (fixed stride 45):
///     [conf, px0, py0, px1, py1, ..., px21, py21]  (22 mouth points, normalized, top-left)
/// Returns face count (≤ max_faces) or -1 on error.
@_cdecl("shim_face_landmarks")
public func shim_face_landmarks(path: UnsafePointer<CChar>?, out: UnsafeMutablePointer<Float>?, max_faces: Int32) -> Int32 {
    guard let cg = loadCGImage(path) else { return -1 }
    guard let out = out, max_faces > 0 else { return 0 }
    let req = VNDetectFaceLandmarksRequest()
    req.usesCPUOnly = false
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try handler.perform([req])
    } catch {
        return -1
    }
    let stride = 45
    var count = 0
    for obs in (req.results ?? []) {
        if count >= Int(max_faces) { break }
        out[count * stride + 0] = Float(obs.confidence)
        // `outerLips` normalizedPoints are image-relative, bottom-left origin.
        let mouth = obs.landmarks?.outerLips?.normalizedPoints ?? []
        for (i, p) in mouth.enumerated() where i < 22 {
            out[count * stride + 1 + i * 2] = Float(p.x)
            out[count * stride + 2 + i * 2] = Float(1.0 - p.y)
        }
        count += 1
    }
    return Int32(count)
}

// MARK: - image loading

private func loadCGImage(_ path: UnsafePointer<CChar>?) -> CGImage? {
    guard let path = path,
          let str = String(cString: path, encoding: .utf8) else { return nil }
    let url = URL(fileURLWithPath: str)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}