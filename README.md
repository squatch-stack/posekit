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

No mesh is generated unless you ask (`--model <detail>`), so a run stops
after image alignment — minutes, not tens of minutes, and no CUDA, no
Homebrew, no Python.

```
swift build -c release
.build/release/posekit ./captures/cannon/images --sequential
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
| `--model <detail>` | Also produce OBJ: `preview`, `reduced`, `medium`, `full`, or `raw` |
| `--preview-model` | Alias for `--model preview` (now OBJ; add `--usdz` for USDZ) |
| `--usdz` | Also produce `model.usdz` at the selected detail; requires `--model` or its alias |
| `--masks <dir>` | Use `<stem>.png` subject masks; skip photos with missing masks and print the count |
| `--high-sensitivity` | Work harder on low-texture scenes |

## Mesh deliverables

Use the same photographs to solve poses and reconstruct a textured mesh:

```sh
swift build -c release
.build/release/posekit ./captures/tree/images --out ./out/tree-mesh \
  --model full --masks ./captures/tree/masks --usdz
```

Replace the relative capture paths with your photograph and mask folders.
`--sequential` is optional when filename order follows a walk around the subject.
The default still exports only poses and the sparse cloud. Mesh requests continue
through geometry reconstruction and texture mapping before the process exits;
a failed or incomplete request returns a nonzero exit status.

| Detail | Intended use |
|---|---|
| `preview` | Fast inspection of alignment and rough shape |
| `reduced` | Smaller mesh for lightweight viewing |
| `medium` | More geometry and texture detail at moderate cost |
| `full` | High-quality deliverable with substantial memory and processing cost |
| `raw` | Highest reconstruction resolution, largest processing and storage cost |

Budget tens of minutes or longer for full/raw captures, versus minutes for pose
alignment; photo count, hardware, scene complexity, and requested detail determine
actual time. These are planning estimates, not measured timings for your capture.
Full and raw output may need simplification for an interactive web gallery.
See Apple's [detail documentation](https://developer.apple.com/documentation/realitykit/photogrammetrysession/request/detail).

The OBJ request targets `<out>/model.obj`; retain the generated MTL and textures
alongside the OBJ. RealityKit documents OBJ output as a folder, so an OS version
may make `model.obj` a directory containing the OBJ and companion assets. The
converter below accepts either the file or that directory (exactly one OBJ).
`--usdz` additionally requests `<out>/model.usdz` at the same detail. Completion
prints each model output path. The current framework's `modelFile` result exposes
only a URL, without triangle statistics; the converter reports actual counts.
The pose files and sparse cloud remain in the same output directory.

### Subject masks

For `images/frame001.HEIC`, supply `masks/frame001.png`: **white is subject,
black is excluded background**. Use unique photograph stems. Each PNG must match
the photograph's exact width, height, crop, and pixel grid in **raw EXIF
orientation**: the stored raster before a viewer rotates or mirrors it for display.
If a masking tool auto-orients the photo, undo that transform on the mask before
saving. Do not resize, crop, rotate, or mirror only one of the pair. The loader
decodes masks into 8-bit grayscale without applying orientation or rescaling;
a dimension mismatch or corrupt mask fails the run. Alignment beyond dimensions
is the operator's responsibility.

`--masks` enables object masking even without `--object`; otherwise the framework
would ignore supplied masks. Photos without a matching PNG are skipped, with an
aggregate count printed before processing. An empty usable set fails. Samples
load lazily to limit memory use, with an explicit sample-ID-to-filename mapping
for pose exports. macOS 15+ preserves the framework-loaded depth, gravity, and
metadata; macOS 14 uses RGB and EXIF metadata only for masked input, so depth-based
scale and gravity alignment are unavailable on that path. Without `--masks`, the
existing folder loader remains in use.
See Apple's [mask requirements](https://developer.apple.com/documentation/realitykit/photogrammetrysample/objectmask).

A canopy of leaves meshes badly: thin overlapping surfaces, occlusion, and leaf
motion make reconstruction unreliable. Static trunks, textured bark, and stone
mesh well by comparison when photographed with sufficient overlap and coverage.
Masks remove background; they cannot repair motion or recover unseen surfaces.

### Single-file GLB for the gallery

After the operator's reconstruction finishes:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements.txt
.venv/bin/python tools/obj_to_glb.py ./out/tree-mesh/model.obj ./out/tree-mesh/model.glb
```

The pinned `trimesh` and `pillow` dependencies load OBJ/MTL, triangulate faces,
and embed the diffuse texture in a single binary glTF file. Conversion prints
source and embedded texture dimensions and byte sizes, vertex count, triangle
count, and final GLB size in bytes. It refuses missing MTLs,
missing/unreadable diffuse textures, and meshes without UVs rather than silently
writing an untextured deliverable. MTL texture-map options are unsupported and must
be baked first; keep the OBJ, MTL, and texture together for loading. This step
packages the mesh; it does not simplify geometry. Upload the `.glb` alone.

### Textures and web delivery

The masked oak run from 169 photographs produced 55,836 vertices and 99,999
triangles: a 10.8 MB OBJ, but a **66.3 MB GLB** when the full colour texture was
embedded. Its companion maps were 72.2 MB colour PNG, 47.7 MB normal PNG,
35.2 MB roughness PNG, 9.5 MB ambient-occlusion PNG, and 137.5 MB displacement
EXR. The textures, rather than the geometry, dominate the gallery's 20 MB budget.
An 8192-square colour map is excessive for this roughly 100,000-triangle web mesh.

The converter now defaults to colour only, a longest side of **2048 pixels**,
and **WebP quality 90**. Resizing preserves aspect ratio with Pillow LANCZOS;
`--max-texture 0` disables resizing but still re-encodes the texture. It prints
source dimensions/bytes, resized dimensions, each embedded image's dimensions/bytes,
and final GLB bytes. A one-line warning appears above 20,000,000 bytes without
refusing to write the file. The actual oak assets are not included here, so run
the conversion to measure its reduced size; no reduced oak size is implied.

```sh
# Smaller web delivery:
.venv/bin/python tools/obj_to_glb.py ./out/tree-mesh/model.obj ./out/tree-mesh/web.glb \
  --max-texture 1024 --texture-quality 85
# Core glTF compatibility and optional maps for Blender:
.venv/bin/python tools/obj_to_glb.py ./out/tree-mesh/model.obj ./out/tree-mesh/detail.glb \
  --texture-format png --extra-maps
```

`--texture-format png|jpeg|webp` selects the encoding. PNG and JPEG are core glTF
formats. WebP declares **EXT_texture_webp** in both `extensionsUsed` and
`extensionsRequired`; **a viewer without that extension will not load it**.
Use JPEG for compact core-compatible output, or PNG for lossless pixels.
`--texture-quality` accepts 0-100 (default 90) for JPEG/WebP and is ignored for
PNG. JPEG discards alpha with a warning; PNG and WebP retain it. If WebP cannot
be encoded or its extension declaration is invalid, the tool reports a JPEG
fallback rather than writing an incorrectly declared WebP asset.

`--extra-maps` adds referenced normal, roughness, and occlusion maps, each with
the same size limit and encoding. Supported MTL keys are `norm`, `map_Kn`,
`map_Bump`/`bump` (assumed tangent-space normal images), `map_Pr`/`map_roughness`,
and `map_AO`/`map_occ`/`map_occlusion`. Scalar roughness is packed into glTF's green
channel with nonmetallic blue; occlusion uses red. PNG is preferable for exact
normal and material values. Height-to-normal conversion is not performed.
Displacement directives are ignored and EXR maps are never embedded: glTF has
no displacement slot. Optional maps are neither required nor embedded by default.

The trunk of a tree meshes well while its canopy does not: a mesh is the right
deliverable for stone, timber and metal, and a splat is right for foliage.

CPU-only verification (no reconstruction session):

```sh
.venv/bin/python -m unittest discover -s tools -p 'test_*.py' -v
.venv/bin/python -m pip install ruff==0.15.5
.venv/bin/ruff check tools --line-length 120 --select E,F,B,RUF
```

The synthetic textured-quad test removes the source assets before reloading the
GLB and checks geometry, UVs, and embedded texture pixels. Additional cases check
missing assets/UVs, the framework's directory-shaped OBJ output, resizing and
encoding, WebP extension references, optional map slots and roughness channels,
displacement exclusion, and the 20 MB warning with an actual oversized GLB.

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
