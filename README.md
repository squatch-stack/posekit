# posekit

Camera poses from Apple's photogrammetry, for splat trainers.

macOS ships a photogrammetry solver that runs on Apple silicon's GPU and
Neural Engine — and since macOS 14 it will hand you the solved camera
poses, not just a mesh. `posekit` is a small CLI that runs
`PhotogrammetrySession` over a folder of images and writes the formats
Gaussian-splat trainers actually consume:

- **`transforms.json`** — nerfstudio layout, per-frame intrinsics,
  camera-to-world matrices (Apple's own convention, passed through
  verbatim);
- **`colmap/sparse/0/`** — COLMAP text model (world-to-camera, OpenCV
  axes), for trainers that read COLMAP;
- **`sparse_pc.ply`** — the solver's sparse point cloud, for splat
  initialization.

No mesh is generated unless you ask (`--preview-model`), so a run stops
after image alignment — minutes, not tens of minutes, and no CUDA, no
Homebrew, no Python.

```
swift build -c release
.build/release/posekit ~/captures/cannon/images --sequential
```

## Why this exists

Every Mac splat pipeline today solves poses with COLMAP on the CPU —
the one stage Apple silicon can't accelerate, because COLMAP's GPU paths
are CUDA-only. Apple's solver runs on the hardware you have. As far as
we can tell this is the first licensed, open-source bridge from
`PhotogrammetrySession` poses to both nerfstudio and COLMAP formats
(one unlicensed macOS-26-only nerfstudio exporter exists in the wild),
with a pre-macOS-26 fallback that synthesizes intrinsics from EXIF.

## Options

| Flag | Meaning |
|---|---|
| `--out <dir>` | Output directory (default: `<input>-poses` beside the input) |
| `--sequential` | Images are a time-ordered walk — a speed hint, not a quality knob |
| `--object` | Single-object capture: keep Apple's automatic masking on. Default is scene mode — masking off, bounding box ignored (macOS 15+) |
| `--preview-model` | Also produce a preview-detail USDZ |
| `--high-sensitivity` | Work harder on low-texture scenes |

## Conventions, and how much to trust them

Apple documents almost nothing about the pose payload's conventions.
What this tool relies on, and its evidence:

- **Camera-to-world, camera looking down −Z** (ARKit/OpenGL axes):
  consistent with Apple's documented conventions everywhere else in the
  ecosystem, and **checked on every invocation** — posekit verifies that
  the solved cameras face the sparse cloud's centroid under this reading
  and warns loudly when they don't.
- **Intrinsics** (`Pose.intrinsics`) exist on **macOS 26+** only,
  ARKit-style column layout, and are per-frame. On macOS 14/15 posekit
  falls back to EXIF 35mm-equivalent focal length with a centered
  principal point, and says so — approximate, but enough for trainers
  that refine poses.
- **Scale** is metric when the images carry depth (iPhone HEIC),
  arbitrary otherwise. **Gravity**: with iPhone metadata present the
  solved frame comes out upright; stripped EXIF makes it arbitrary.
- Poses cover **registered images only**; the report lists the count
  and the rejects. ~50% registration on hard handheld sets is normal.
- Lens distortion (`lensDistortionData`, macOS 26+) is a radial lookup
  table, not Brown–Conrady coefficients; posekit currently writes zero
  distortion (fitting k1/k2 to the LUT is on the roadmap).
- The COLMAP `points3D.txt` carries no reprojection errors or tracks —
  Apple doesn't expose them. It seeds splat trainers fine; it cannot
  restart a COLMAP bundle adjustment.

## Limits

Read at runtime from `PhotogrammetrySession.limits` and printed on every
run (a 64 GB M-series Mac reports 2000 images / 16384 px). Excess images
are rejected as `invalidSample`, not silently dropped by posekit.

## License

Apache-2.0. Built by [Squatch Stack](https://github.com/squatch-stack).
