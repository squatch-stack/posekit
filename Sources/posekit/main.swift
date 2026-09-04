// posekit — camera poses from Apple's photogrammetry, for splat trainers.
//
//     posekit <image-folder> [--out <dir>] [--sequential] [--object]
//             [--model <detail>] [--usdz] [--masks <dir>] [--high-sensitivity]
//
// Runs RealityKit's PhotogrammetrySession over a folder of images and
// exports what Gaussian-splat trainers eat: a nerfstudio transforms.json
// (camera-to-world, OpenGL camera axes — Apple's pose convention verbatim),
// a COLMAP text model (world-to-camera, OpenCV axes), and the sparse point
// cloud as PLY. No mesh is requested unless --model is passed, so
// the run stops after image alignment — the cheap part.
//
// Conventions, verified against the macOS 26 SDK and validated empirically
// (see README):
//   - Pose is camera-to-world; camera looks down -Z (ARKit/OpenGL axes).
//   - posesBySample/urlsBySample are keyed by sample id; for folder input
//     urlsBySample is the ONLY reliable join back to filenames.
//   - Pose.intrinsics and lensDistortionData exist on macOS 26+ only; on
//     macOS 14/15 intrinsics are synthesized from EXIF 35mm-equivalent
//     focal length and flagged approximate.
//   - Poses cover only registered samples; the report says which were not.

import Foundation
import ImageIO
import CoreVideo
import RealityKit
import simd

// MARK: - Options

struct Options {
    var input: URL
    var out: URL
    var sequential = false
    var objectMode = false
    var modelDetail: PhotogrammetrySession.Request.Detail?
    var usdz = false
    var masks: URL?
    var highSensitivity = false
}

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("posekit: \(message)\n".utf8))
    exit(code)
}

func parseOptions() -> Options {
    let args = Array(CommandLine.arguments.dropFirst())
    var inputArg: String?
    var outArg: String?
    var options = Options(input: URL(fileURLWithPath: "."), out: URL(fileURLWithPath: "."))
    var i = 0
    func value() -> String {
        guard i + 1 < args.count, !args[i + 1].hasPrefix("--") else {
            fail("\(args[i]) requires a value", code: 2)
        }
        i += 1
        return args[i]
    }
    let details: [String: PhotogrammetrySession.Request.Detail] = [
        "preview": .preview, "reduced": .reduced, "medium": .medium, "full": .full, "raw": .raw,
    ]
    while i < args.count {
        switch args[i] {
        case "--out": outArg = value()
        case "--masks": options.masks = URL(fileURLWithPath: value(), isDirectory: true)
        case "--sequential": options.sequential = true
        case "--object": options.objectMode = true
        case "--high-sensitivity": options.highSensitivity = true
        case "--usdz": options.usdz = true
        case "--model", "--preview-model":
            let name = args[i] == "--preview-model" ? "preview" : value()
            guard let detail = details[name] else {
                fail("invalid model detail '\(name)' (preview, reduced, medium, full, raw)", code: 2)
            }
            if let previous = options.modelDetail, previous != detail {
                fail("conflicting model detail options", code: 2)
            }
            options.modelDetail = detail
        case "--help", "-h":
            print("""
            usage: posekit <image-folder> [--out <dir>] [--sequential] [--object]
                           [--model <preview|reduced|medium|full|raw>] [--usdz]
                           [--masks <dir>] [--high-sensitivity]
              --preview-model     alias for --model preview (OBJ output)
              --usdz              also write model.usdz; requires --model
              --masks <dir>       <stem>.png, white subject, raw photo pixel grid
              --object            enable automatic object masking
              --sequential        time-ordered images (speed hint)
              --high-sensitivity  work harder on low-texture scenes
            """)
            exit(0)
        default:
            guard !args[i].hasPrefix("-"), inputArg == nil else {
                fail("unexpected argument '\(args[i])'; use --help", code: 2)
            }
            inputArg = args[i]
        }
        i += 1
    }
    guard let inputArg else { fail("image folder required; use --help", code: 2) }
    guard !options.usdz || options.modelDetail != nil else {
        fail("--usdz requires --model (or --preview-model)", code: 2)
    }
    options.input = URL(fileURLWithPath: inputArg, isDirectory: true)
    options.out = outArg.map { URL(fileURLWithPath: $0, isDirectory: true) }
        ?? options.input.deletingLastPathComponent()
            .appendingPathComponent(options.input.lastPathComponent + "-poses")
    return options
}

// Decode the stored pixel grid without applying EXIF orientation, scaling, or cropping.
func pixelBuffer(at url: URL, grayscale: Bool) throws -> CVPixelBuffer {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "posekit", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Cannot decode \(url.lastPathComponent)"])
    }
    var buffer: CVPixelBuffer?
    let format = grayscale ? kCVPixelFormatType_OneComponent8 : kCVPixelFormatType_32ARGB
    let status = CVPixelBufferCreate(kCFAllocatorDefault, image.width, image.height, format,
                                    nil, &buffer)
    guard status == kCVReturnSuccess, let buffer else {
        throw NSError(domain: "posekit", code: Int(status))
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let space = grayscale ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB()
    let alpha = grayscale ? CGImageAlphaInfo.none : CGImageAlphaInfo.noneSkipFirst
    guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: image.width, height: image.height, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: space, bitmapInfo: alpha.rawValue) else {
        throw NSError(domain: "posekit", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Cannot create image context"])
    }
    context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return buffer
}

func maskedSample(id: Int, photo: URL, mask: URL) throws -> PhotogrammetrySample {
    var sample: PhotogrammetrySample
    if #available(macOS 15.0, *) {
        // Preserve the framework's depth, gravity, and EXIF decoding, but assign a
        // stable ID so sequence input can still be joined to the source photo.
        let loaded = try PhotogrammetrySample(contentsOf: photo)
        sample = PhotogrammetrySample(id: id, image: loaded.image)
        sample.metadata = loaded.metadata
        sample.depthDataMap = loaded.depthDataMap
        sample.gravity = loaded.gravity
    } else {
        sample = PhotogrammetrySample(id: id, image: try pixelBuffer(at: photo, grayscale: false))
        if let source = CGImageSourceCreateWithURL(photo as CFURL, nil),
           let metadata = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] {
            sample.metadata = metadata
        }
    }
    let buffer = try pixelBuffer(at: mask, grayscale: true)
    guard CVPixelBufferGetWidth(buffer) == CVPixelBufferGetWidth(sample.image),
          CVPixelBufferGetHeight(buffer) == CVPixelBufferGetHeight(sample.image) else {
        throw NSError(domain: "posekit", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Mask dimensions differ for \(photo.lastPathComponent)"])
    }
    sample.objectMask = buffer
    return sample
}

// MARK: - Image metadata (dimensions always; EXIF focal fallback pre-26)

struct ImageInfo {
    var width: Int
    var height: Int
    /// 35 mm-equivalent focal length, when EXIF carries it.
    var focal35: Double?
}

func imageInfo(at url: URL) -> ImageInfo? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
            as? [CFString: Any],
        let width = props[kCGImagePropertyPixelWidth] as? Int,
        let height = props[kCGImagePropertyPixelHeight] as? Int
    else { return nil }
    var focal35: Double?
    if let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] {
        focal35 = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? Double
    }
    return ImageInfo(width: width, height: height, focal35: focal35)
}

// MARK: - Intrinsics

struct Intrinsics {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
    /// False when synthesized from EXIF rather than solved by Apple.
    var solved: Bool
}

func intrinsics(for pose: PhotogrammetrySession.Pose, info: ImageInfo) -> Intrinsics? {
    if #available(macOS 26.0, *) {
        if let k = pose.intrinsics {
            // Column-major, ARKit layout: fx = [0][0], fy = [1][1],
            // principal point in column 2.
            return Intrinsics(
                fx: Double(k.columns.0.x), fy: Double(k.columns.1.y),
                cx: Double(k.columns.2.x), cy: Double(k.columns.2.y),
                solved: true)
        }
    }
    guard let f35 = info.focal35 else { return nil }
    // 35mm-equivalent focal length spans a 36 mm frame width; the principal
    // point is assumed central. Approximate, and marked as such.
    let fx = Double(info.width) * f35 / 36.0
    return Intrinsics(
        fx: fx, fy: fx,
        cx: Double(info.width) / 2, cy: Double(info.height) / 2,
        solved: false)
}

// MARK: - Session

let options = parseOptions()

guard PhotogrammetrySession.isSupported else {
    FileHandle.standardError.write(
        Data("posekit: PhotogrammetrySession is not supported on this machine\n".utf8))
    exit(1)
}

var configuration = PhotogrammetrySession.Configuration()
configuration.sampleOrdering = options.sequential ? .sequential : .unordered
configuration.featureSensitivity = options.highSensitivity ? .high : .normal
configuration.isObjectMaskingEnabled = options.objectMode || options.masks != nil
if #available(macOS 15.0, *), !options.objectMode, options.masks == nil {
    // Scene mode: recover everything the images saw, not one masked object.
    configuration.ignoreBoundingBox = true
}

let limits = PhotogrammetrySession.limits
print("posekit: limits on this machine — \(limits.maximumNumberOfInputImages) images, "
    + "\(limits.maximumInputImageDimension) px")

var sampleURLs: [Int: URL] = [:]
var missingMasks = 0
let session: PhotogrammetrySession
if let masks = options.masks {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: masks.path, isDirectory: &isDirectory),
          isDirectory.boolValue else { fail("mask directory does not exist") }
    let photos = try FileManager.default.contentsOfDirectory(
        at: options.input, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
        .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
            && imageInfo(at: $0) != nil }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    var entries: [(id: Int, photo: URL, mask: URL)] = []
    for (index, photo) in photos.enumerated() {
        let mask = masks.appendingPathComponent(photo.deletingPathExtension().lastPathComponent + ".png")
        guard FileManager.default.fileExists(atPath: mask.path) else {
            missingMasks += 1
            continue
        }
        let id = index + 1
        sampleURLs[id] = photo
        entries.append((id, photo, mask))
    }
    print("posekit: masks — \(entries.count) photos included, \(missingMasks) skipped for missing masks")
    guard !entries.isEmpty else { fail("no photos with matching masks") }
    // Decode lazily: a full capture must not retain every uncompressed photo in RAM.
    let samples = entries.lazy.map { entry -> PhotogrammetrySample in
        do { return try maskedSample(id: entry.id, photo: entry.photo, mask: entry.mask) }
        catch { fail("cannot load masked photo \(entry.photo.lastPathComponent): \(error.localizedDescription)") }
    }
    session = try PhotogrammetrySession(input: samples, configuration: configuration)
} else {
    session = try PhotogrammetrySession(input: options.input, configuration: configuration)
}

var requests: [PhotogrammetrySession.Request] = [.poses, .pointCloud]
if let detail = options.modelDetail {
    requests.append(.modelFile(url: options.out.appendingPathComponent("model.obj"), detail: detail))
    if options.usdz {
        requests.append(.modelFile(url: options.out.appendingPathComponent("model.usdz"), detail: detail))
    }
}
var pendingRequests = Set(requests)
var requestFailed = false

try FileManager.default.createDirectory(at: options.out, withIntermediateDirectories: true)

var poses: PhotogrammetrySession.Poses?
var cloud: PhotogrammetrySession.PointCloud?
var rejected: [(id: Int, reason: String)] = []
var downsampled = false
let started = Date()

try session.process(requests: requests)
outputLoop: for try await output in session.outputs {
    switch output {
    case .requestComplete(let request, let result):
        pendingRequests.remove(request)
        switch result {
        case .poses(let p):
            poses = p
            print(String(format: "posekit: poses in %.0f s — %d registered",
                         Date().timeIntervalSince(started), p.posesBySample.count))
        case .pointCloud(let c):
            cloud = c
            print("posekit: sparse cloud — \(c.points.count) points")
        case .modelFile(let url):
            print("posekit: model — \(url.path)")
            // The modelFile result contains only a URL, not a triangle count.
            print("posekit: triangle count not reported by framework; GLB conversion reports it")
        default:
            break
        }
        // Everything wanted has arrived; a mesh nobody asked for should not
        // keep the GPU warm.
        if poses != nil, cloud != nil, options.modelDetail == nil {
            session.cancel()
        }
    case .requestError(let request, let error):
        requestFailed = true
        FileHandle.standardError.write(
            Data("posekit: request \(request) failed — \(error)\n".utf8))
    case .invalidSample(let id, let reason):
        rejected.append((id, reason))
    case .skippedSample(let id):
        rejected.append((id, "skipped"))
    case .automaticDownsampling:
        downsampled = true
    case .processingComplete, .processingCancelled:
        break outputLoop
    default:
        break
    }
}

guard !requestFailed, pendingRequests.isEmpty else {
    fail("one or more requested outputs were not produced")
}

guard let poses else {
    FileHandle.standardError.write(Data("posekit: no poses were produced\n".utf8))
    exit(1)
}

// MARK: - Join, validate, report

struct Frame {
    var id: Int
    var url: URL
    var pose: PhotogrammetrySession.Pose
    var info: ImageInfo
    var k: Intrinsics
}

var frames: [Frame] = []
var missingIntrinsics = 0
for (id, pose) in poses.posesBySample.sorted(by: { $0.key < $1.key }) {
    guard let url = sampleURLs[id] ?? poses.urlsBySample[id] else { continue }
    guard let info = imageInfo(at: url) else { continue }
    guard let k = intrinsics(for: pose, info: info) else {
        missingIntrinsics += 1
        continue
    }
    frames.append(Frame(id: id, url: url, pose: pose, info: info, k: k))
}

let total = (try? FileManager.default.contentsOfDirectory(
    at: options.input, includingPropertiesForKeys: nil)
    .filter { !$0.hasDirectoryPath }.count) ?? frames.count

// The convention self-check: under camera-to-world with a -Z forward axis,
// nearly every registered camera should face the sparse cloud's centroid.
// This is the test that distinguishes the documented-nowhere convention
// from its alternatives, run on every invocation rather than trusted once.
var facingFraction = -1.0
if let cloud, !cloud.points.isEmpty, !frames.isEmpty {
    var centroid = SIMD3<Float>(0, 0, 0)
    for p in cloud.points { centroid += p.position }
    centroid /= Float(cloud.points.count)
    var facing = 0
    for f in frames {
        let r = simd_float3x3(f.pose.rotation)
        let forward = -r.columns.2
        if simd_dot(forward, centroid - f.pose.translation) > 0 { facing += 1 }
    }
    facingFraction = Double(facing) / Double(frames.count)
}

print("posekit: \(frames.count)/\(total) frames registered"
    + (rejected.isEmpty ? "" : " (\(rejected.count) rejected)")
    + (missingIntrinsics > 0
        ? "; \(missingIntrinsics) dropped for missing intrinsics/EXIF" : ""))
if facingFraction >= 0 {
    print(String(format: "posekit: %.0f%% of cameras face the cloud centroid",
                 facingFraction * 100)
        + (facingFraction < 0.8 ? "  ← LOW: pose convention or capture suspect" : ""))
}
if downsampled {
    print("posekit: WARNING — automatic downsampling occurred; intrinsics may "
        + "reference reduced resolutions")
}
if let first = frames.first, !first.k.solved {
    print("posekit: intrinsics synthesized from EXIF (pre-macOS-26 fallback) — "
        + "approximate focal, centered principal point")
}

// MARK: - Writers

func writeSparsePLY(_ cloud: PhotogrammetrySession.PointCloud, to url: URL) throws {
    var header = """
    ply
    format binary_little_endian 1.0
    element vertex \(cloud.points.count)
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header

    """
    header.removeLast()
    var data = Data(header.utf8)
    for p in cloud.points {
        var xyz = [p.position.x, p.position.y, p.position.z]
        data.append(Data(bytes: &xyz, count: 12))
        data.append(contentsOf: [p.color.x, p.color.y, p.color.z])
    }
    try data.write(to: url)
}

func writeTransformsJSON(to url: URL, plyName: String?) throws {
    func matrixRows(_ pose: PhotogrammetrySession.Pose) -> [[Double]] {
        let r = simd_float3x3(pose.rotation)
        let t = pose.translation
        return [
            [Double(r.columns.0.x), Double(r.columns.1.x), Double(r.columns.2.x), Double(t.x)],
            [Double(r.columns.0.y), Double(r.columns.1.y), Double(r.columns.2.y), Double(t.y)],
            [Double(r.columns.0.z), Double(r.columns.1.z), Double(r.columns.2.z), Double(t.z)],
            [0, 0, 0, 1],
        ]
    }
    var doc: [String: Any] = ["camera_model": "OPENCV"]
    if let plyName { doc["ply_file_path"] = plyName }
    doc["frames"] = frames.map { f -> [String: Any] in
        [
            "file_path": relativePath(of: f.url, from: options.out),
            "w": f.info.width, "h": f.info.height,
            "fl_x": f.k.fx, "fl_y": f.k.fy, "cx": f.k.cx, "cy": f.k.cy,
            "k1": 0, "k2": 0, "p1": 0, "p2": 0,
            "transform_matrix": matrixRows(f.pose),
        ]
    }
    let json = try JSONSerialization.data(
        withJSONObject: doc, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: url)
}

func relativePath(of target: URL, from base: URL) -> String {
    let t = target.standardizedFileURL.pathComponents
    let b = base.standardizedFileURL.pathComponents
    var i = 0
    while i < min(t.count, b.count), t[i] == b[i] { i += 1 }
    return String(repeating: "../", count: b.count - i) + t[i...].joined(separator: "/")
}

func writeCOLMAP(to dir: URL, cloud: PhotogrammetrySession.PointCloud?) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    var cameras = "# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
    var images = "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
    for (index, f) in frames.enumerated() {
        let id = index + 1
        cameras += "\(id) PINHOLE \(f.info.width) \(f.info.height) "
            + "\(f.k.fx) \(f.k.fy) \(f.k.cx) \(f.k.cy)\n"

        // Apple: camera-to-world, OpenGL axes (-Z forward). COLMAP wants
        // world-to-camera, OpenCV axes (+Z forward, y down). Flip the y and
        // z basis vectors, then invert the rigid transform.
        var r = simd_float3x3(f.pose.rotation)
        r.columns.1 = -r.columns.1
        r.columns.2 = -r.columns.2
        let rwc = r.transpose
        let twc = -(rwc * f.pose.translation)
        let q = simd_quatf(rwc)
        images += "\(id) \(q.real) \(q.imag.x) \(q.imag.y) \(q.imag.z) "
            + "\(twc.x) \(twc.y) \(twc.z) \(id) \(f.url.lastPathComponent)\n\n"
    }
    try cameras.write(
        to: dir.appendingPathComponent("cameras.txt"), atomically: true, encoding: .utf8)
    try images.write(
        to: dir.appendingPathComponent("images.txt"), atomically: true, encoding: .utf8)

    var points = "# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
    if let cloud {
        // No reprojection errors or tracks exist in Apple's payload; zeros
        // and empty tracks are written, which seeds trainers but cannot
        // restart a COLMAP bundle adjustment.
        for (i, p) in cloud.points.enumerated() {
            points += "\(i + 1) \(p.position.x) \(p.position.y) \(p.position.z) "
                + "\(p.color.x) \(p.color.y) \(p.color.z) 0\n"
        }
    }
    try points.write(
        to: dir.appendingPathComponent("points3D.txt"), atomically: true, encoding: .utf8)
}

var plyName: String? = nil
if let cloud {
    plyName = "sparse_pc.ply"
    try writeSparsePLY(cloud, to: options.out.appendingPathComponent("sparse_pc.ply"))
}
try writeTransformsJSON(
    to: options.out.appendingPathComponent("transforms.json"), plyName: plyName)
try writeCOLMAP(
    to: options.out.appendingPathComponent("colmap/sparse/0"), cloud: cloud)

print("posekit: wrote \(options.out.path)/{transforms.json, colmap/sparse/0, "
    + (plyName != nil ? "sparse_pc.ply" : "") + "}")
