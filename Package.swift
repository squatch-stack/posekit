// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "posekit",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "posekit", path: "Sources/posekit")
    ]
)
