#!/usr/bin/env python3
"""Convert a textured OBJ bundle to one GLB, failing instead of losing textures."""

import argparse
import json
from pathlib import Path
import shlex
import struct

from PIL import Image
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


def check_textures(source: Path) -> None:
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
                materials[current] = None
            elif key.lower() == "map_kd" and current is not None:
                # trimesh does not interpret MTL map options; reject instead of misrendering.
                if value.startswith("-"):
                    raise ValueError("MTL map_Kd options are unsupported; bake them into the texture first")
                texture = library.parent / value
                if not texture.is_file():
                    raise ValueError(f"Missing texture: {texture}")
                with Image.open(texture) as image:
                    image.verify()
                materials[current] = texture
    for name in sorted(used):
        if materials.get(name) is None:
            raise ValueError(f"Missing diffuse texture (map_Kd) for material {name!r}")


def convert(source: Path, destination: Path) -> tuple[int, int, int]:
    source = obj_path(source)
    if destination.suffix.lower() != ".glb":
        raise ValueError("Output must have a .glb extension")
    check_textures(source)
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
        image.load()
        vertices += len(mesh.vertices)
        triangles += len(mesh.faces)
    data = trimesh.exchange.gltf.export_glb(scene)
    # Check the actual container: all image data must live in GLB buffer views.
    json_size = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + json_size])
    if not document.get("images") or any("bufferView" not in image for image in document["images"]):
        raise ValueError("Exporter did not embed the textures")
    if any("uri" in buffer for buffer in document.get("buffers", [])):
        raise ValueError("Exporter emitted an external buffer")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print(f"{destination}: {vertices:,} vertices, {triangles:,} triangles, {len(data):,} bytes GLB")
    return vertices, triangles, len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OBJ file (or RealityKit OBJ output directory)")
    parser.add_argument("output", type=Path, help="Destination .glb file")
    args = parser.parse_args()
    try:
        convert(args.input, args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"obj_to_glb: {error}\n")


if __name__ == "__main__":
    main()
