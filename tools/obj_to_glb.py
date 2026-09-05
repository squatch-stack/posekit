#!/usr/bin/env python3
"""Convert a textured OBJ bundle to one GLB, failing instead of losing textures."""

import argparse
import json
from io import BytesIO
from pathlib import Path
import shlex
import struct

from PIL import Image, features
import trimesh


def obj_path(source: Path) -> Path:
    """RealityKit can return an OBJ output directory rather than a single file."""
    if source.is_dir():
        candidates = sorted(source.rglob("*.obj"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one OBJ inside {source}, found {len(candidates)}")
        return candidates[0]
    if source.suffix.lower() != ".obj":
        raise ValueError("Input must be an OBJ file or a directory containing one OBJ")
    return source


MAP_SLOTS = {
    "map_kd": "colour",
    "norm": "normal",
    "map_kn": "normal",
    "map_bump": "normal",
    "bump": "normal",
    "map_pr": "roughness",
    "map_roughness": "roughness",
    "map_ao": "occlusion",
    "map_occ": "occlusion",
    "map_occlusion": "occlusion",
}
WEB_BUDGET = 20_000_000


def check_textures(source: Path, extra_maps: bool = False) -> dict[str, dict[str, Path]]:
    """Check references before trimesh's permissive loader can hide missing assets."""
    libraries = []
    used = set()
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        key, value = fields
        if key == "mtllib":
            # Accept both a filename with spaces and multiple library filenames.
            names = [value] if (source.parent / value).is_file() else shlex.split(value, comments=True)
            libraries.extend(source.parent / name for name in names)
        elif key == "usemtl":
            used.add(value)
    if not libraries or not used:
        raise ValueError("OBJ must reference an MTL library and use a textured material")
    materials = {}
    for library in libraries:
        current = None
        for line in library.read_text(encoding="utf-8-sig").splitlines():
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                continue
            key, value = fields
            if key.lower() == "newmtl":
                current = value
                materials[current] = {}
            elif current is not None and key.lower() in MAP_SLOTS:
                slot = MAP_SLOTS[key.lower()]
                if slot != "colour" and not extra_maps:
                    continue
                # Height/displacement maps have no glTF slot and are never opened.
                if value.startswith("-"):
                    raise ValueError(f"MTL {key} options are unsupported; bake them into the texture first")
                texture = library.parent / value
                if texture.suffix.lower() == ".exr":
                    if slot == "colour":
                        raise ValueError("EXR diffuse textures are unsupported")
                    continue
                materials[current][slot] = texture
    for name in sorted(used):
        if not materials.get(name, {}).get("colour"):
            raise ValueError(f"Missing diffuse texture (map_Kd) for material {name!r}")
        for texture in materials[name].values():
            if not texture.is_file():
                raise ValueError(f"Missing texture: {texture}")
            with Image.open(texture) as image:
                image.verify()
    return {name: materials[name] for name in used}


def prepare_texture(path: Path, slot: str, max_texture: int) -> Image.Image:
    with Image.open(path) as original:
        original.load()
        image = original.convert("RGBA" if "A" in original.getbands() or "transparency" in original.info else "RGB")
        if max_texture:
            image.thumbnail((max_texture, max_texture), Image.Resampling.LANCZOS)
        print(
            f"Texture {path.name} ({slot}): {original.width}x{original.height}, "
            f"{path.stat().st_size:,} bytes source -> {image.width}x{image.height}"
        )
    if slot == "roughness":
        # glTF reads roughness from G and metallic from B; MTL Pr is scalar.
        roughness = image.convert("L")
        image = Image.merge("RGB", (Image.new("L", image.size, 255), roughness, Image.new("L", image.size, 0)))
    return image


def export_textures(scene: trimesh.Scene, texture_format: str, quality: int) -> bytes:
    def encode_images(buffers, tree):
        # Trimesh first writes lossless PNG. Re-encode through its buffer hook so
        # Pillow's requested quality is honored without a second lossy encode.
        keys = list(buffers)
        encoded_views = set()
        for index, entry in enumerate(tree["images"]):
            view = entry["bufferView"]
            if view not in encoded_views:
                key = keys[view]
                with Image.open(BytesIO(buffers[key])) as original:
                    image = original.convert("RGB") if texture_format == "jpeg" else original
                    if texture_format == "jpeg" and "A" in original.getbands():
                        print("Warning: JPEG discards texture alpha; use --texture-format png or webp to retain it.")
                    output = BytesIO()
                    options = {"quality": quality} if texture_format != "png" else {}
                    image.save(output, format=texture_format.upper(), **options)
                    data = output.getvalue()
                    print(
                        f"Embedded texture {index}: {image.width}x{image.height}, "
                        f"{len(data):,} bytes {texture_format.upper()}"
                    )
                buffers[key] = data + b"\0" * (-len(data) % 4)
                encoded_views.add(view)
            entry["mimeType"] = f"image/{texture_format}"
        if texture_format == "webp":
            for texture in tree["textures"]:
                texture.setdefault("extensions", {})["EXT_texture_webp"] = {"source": texture.pop("source")}
            for field in ("extensionsUsed", "extensionsRequired"):
                tree.setdefault(field, []).append("EXT_texture_webp")

    return trimesh.exchange.gltf.export_glb(scene, buffer_postprocessor=encode_images)


def convert(
    source: Path, destination: Path, max_texture: int = 2048,
    texture_format: str = "webp", texture_quality: int = 90, extra_maps: bool = False,
) -> tuple[int, int, int]:
    if max_texture < 0:
        raise ValueError("--max-texture must be nonnegative")
    if texture_format not in {"png", "jpeg", "webp"}:
        raise ValueError("--texture-format must be png, jpeg or webp")
    if not 0 <= texture_quality <= 100:
        raise ValueError("--texture-quality must be between 0 and 100")
    if texture_format == "webp" and not features.check("webp"):
        print("Warning: WebP encoding unavailable; falling back to JPEG.")
        texture_format = "jpeg"
    source = obj_path(source)
    if destination.suffix.lower() != ".glb":
        raise ValueError("Output must have a .glb extension")
    references = check_textures(source, extra_maps)
    prepared = {}
    scene = trimesh.load_scene(source, process=False)
    if not scene.geometry:
        raise ValueError("OBJ contains no mesh geometry")
    vertices = triangles = 0
    for mesh in scene.geometry.values():
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("OBJ contains empty or non-mesh geometry")
        visual = mesh.visual
        material = getattr(visual, "material", None)
        image = getattr(material, "image", None)
        if visual.kind != "texture" or visual.uv is None or len(visual.uv) != len(mesh.vertices) or image is None:
            raise ValueError("Every mesh must have UV coordinates and a loaded diffuse texture")
        if material.name not in prepared:
            maps = references.get(material.name)
            if maps is None:
                raise ValueError(f"Loaded material {material.name!r} does not match the MTL")
            pbr = material.to_pbr()
            slots = {
                "colour": "baseColorTexture", "normal": "normalTexture",
                "roughness": "metallicRoughnessTexture", "occlusion": "occlusionTexture",
            }
            for slot, path in maps.items():
                setattr(pbr, slots[slot], prepare_texture(path, slot, max_texture))
            if "roughness" in maps:
                pbr.roughnessFactor = 1.0
                pbr.metallicFactor = 0.0
            prepared[material.name] = pbr
        visual.material = prepared[material.name]
        vertices += len(mesh.vertices)
        triangles += len(mesh.faces)
    data = export_textures(scene, texture_format, texture_quality)
    # Check the actual container: all image data must live in GLB buffer views.
    json_size = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + json_size])
    if not document.get("images") or any("bufferView" not in image for image in document["images"]):
        raise ValueError("Exporter did not embed the textures")
    if any("uri" in buffer for buffer in document.get("buffers", [])):
        raise ValueError("Exporter emitted an external buffer")
    if texture_format == "webp":
        valid = all("EXT_texture_webp" in document.get(field, []) for field in (
            "extensionsUsed", "extensionsRequired"
        )) and all(
            texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source") is not None
            for texture in document.get("textures", [])
        )
        if not valid:
            print("Warning: exporter did not write EXT_texture_webp correctly; falling back to JPEG.")
            data = export_textures(scene, "jpeg", texture_quality)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print(f"{destination}: {vertices:,} vertices, {triangles:,} triangles, {len(data):,} bytes GLB")
    if len(data) > WEB_BUDGET:
        print("Warning: GLB exceeds 20,000,000 bytes; use a smaller --max-texture to shrink it further.")
    return vertices, triangles, len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OBJ file (or RealityKit OBJ output directory)")
    parser.add_argument("output", type=Path, help="Destination .glb file")
    parser.add_argument("--max-texture", type=int, default=2048,
                        help="Maximum texture side in pixels (default: 2048; 0 disables resizing)")
    parser.add_argument("--texture-format", choices=("png", "jpeg", "webp"), default="webp",
                        help="Encoding (default: webp). WebP requires EXT_texture_webp; a viewer without "
                             "that extension will not load it. PNG and JPEG work in core glTF.")
    parser.add_argument("--texture-quality", type=int, default=90,
                        help="JPEG/WebP quality, 0-100 (default: 90); ignored for PNG")
    parser.add_argument("--extra-maps", action="store_true",
                        help="Include referenced normal, roughness and occlusion maps; never displacement/EXR")
    args = parser.parse_args()
    try:
        convert(args.input, args.output, max_texture=args.max_texture, texture_format=args.texture_format,
                texture_quality=args.texture_quality, extra_maps=args.extra_maps)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"obj_to_glb: {error}\n")


if __name__ == "__main__":
    main()
