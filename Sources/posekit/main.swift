// posekit — camera poses from Apple's photogrammetry, for splat trainers.
//
//     posekit <image-folder> [--out <dir>] [--sequential] [--object]
//             [--preview-model] [--high-sensitivity]
//
// Runs RealityKit's PhotogrammetrySession over a folder of images and
// exports what Gaussian-splat trainers eat: a nerfstudio transforms.json
// (camera-to-world, OpenGL camera axes — Apple's pose convention verbatim),
// a COLMAP text model (world-to-camera, OpenCV axes), and the sparse point
// cloud as PLY. No mesh is requested unless --preview-model is passed, so
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
import RealityKit
import simd

// MARK: - Options

struct Options {
    var input: URL
    var out: URL
    var sequential = false
    var objectMode = false
    var previewModel = false
    var highSensitivity = false
}

func parseOptions() -> Options {
    var args = Array(CommandLine.arguments.dropFirst())
    func flag(_ name: String) -> Bool {
        if let i = args.firstIndex(of: name) {
            args.remove(at: i)
            return true
        }
        return false
    }
    func value(_ name: String) -> String? {
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        let v = args[i + 1]
        args.removeSubrange(i...(i + 1))
        return v
    }
    let sequential = flag("--sequential")
    let objectMode = flag("--object")
    let previewModel = flag("--preview-model")
    let highSensitivity = flag("--high-sensitivity")
    let outArg = value("--out")

    guard args.count == 1 else {
        FileHandle.standardError.write(Data("""
        posekit — camera poses from Apple's photogrammetry, for splat trainers

        usage: posekit <image-folder> [--out <dir>] [--sequential] [--object]
                       [--preview-model] [--high-sensitivity]

          --sequential        images are a time-ordered walk (speed hint only)
          --object            single-object capture: keep Apple's auto-masking
                              (default is scene mode: masking off, and the
                              bounding box ignored where the OS allows)
          --preview-model     also produce a preview-detail USDZ
          --high-sensitivity  work harder on low-texture scenes

        """.utf8))
        exit(2)
    }
    let input = URL(fileURLWithPath: args[0], isDirectory: true)
    let out = outArg.map { URL(fileURLWithPath: $0, isDirectory: true) }
        ?? input.deletingLastPathComponent()
            .appendingPathComponent(input.lastPathComponent + "-poses")
    return Options(
        input: input, out: out, sequential: sequential, objectMode: objectMode,
        previewModel: previewModel, highSensitivity: highSensitivity)
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
configuration.isObjectMaskingEnabled = options.objectMode
if #available(macOS 15.0, *), !options.objectMode {
    // Scene mode: recover everything the images saw, not one masked object.
    configuration.ignoreBoundingBox = true
}

let limits = PhotogrammetrySession.limits
print("posekit: limits on this machine — \(limits.maximumNumberOfInputImages) images, "
    + "\(limits.maximumInputImageDimension) px")

let session = try PhotogrammetrySession(input: options.input, configuration: configuration)

var requests: [PhotogrammetrySession.Request] = [.poses, .pointCloud]
let modelURL = options.out.appendingPathComponent("preview.usdz")
if options.previewModel {
    requests.append(.modelFile(url: modelURL, detail: .preview))
}

try FileManager.default.createDirectory(at: options.out, withIntermediateDirectories: true)

var poses: PhotogrammetrySession.Poses?
var cloud: PhotogrammetrySession.PointCloud?
var rejected: [(id: Int, reason: String)] = []
var downsampled = false
let started = Date()

try session.process(requests: requests)
for try await output in session.outputs {
    switch output {
    case .requestComplete(let request, let result):
        switch result {
        case .poses(let p):
            poses = p
            print(String(format: "posekit: poses in %.0f s — %d registered",
                         Date().timeIntervalSince(started), p.posesBySample.count))
        case .pointCloud(let c):
            cloud = c
            print("posekit: sparse cloud — \(c.points.count) points")
        case .modelFile(let url):
            print("posekit: preview model — \(url.lastPathComponent)")
        default:
            break
        }
        // Everything wanted has arrived; a mesh nobody asked for should not
        // keep the GPU warm.
        if poses != nil, cloud != nil, !options.previewModel {
            session.cancel()
        }
        _ = request
    case .requestError(let request, let error):
        FileHandle.standardError.write(
            Data("posekit: request \(request) failed — \(error)\n".utf8))
    case .invalidSample(let id, let reason):
        rejected.append((id, reason))
    case .skippedSample(let id):
        rejected.append((id, "skipped"))
    case .automaticDownsampling:
        downsampled = true
    case .processingComplete, .processingCancelled:
        break
    default:
        break
    }
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
    guard let url = poses.urlsBySample[id] else { continue }
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
