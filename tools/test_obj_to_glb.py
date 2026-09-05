"""CPU-only conversion tests; never run a photogrammetry session."""

from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import obj_to_glb

import numpy as np
from PIL import Image
import trimesh


CONVERTER = Path(__file__).with_name("obj_to_glb.py")


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.obj = self.root / "quad.obj"
        self.glb = self.root / "quad.glb"
        self.obj.write_text(
            "mtllib quad.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "usemtl painted\nf 1/1 2/2 3/3 4/4\n"
        )
        (self.root / "quad.mtl").write_text("newmtl painted\nKd 1 1 1\nmap_Kd texture.png\n")
        self.pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
        Image.fromarray(self.pixels).save(self.root / "texture.png")

    def run_converter(self, source=None, *options):
        return subprocess.run(
            [sys.executable, str(CONVERTER), str(source or self.obj), str(self.glb), *options],
            capture_output=True, text=True, check=False,
        )

    def test_textured_quad_round_trip(self):
        result = self.run_converter(None, "--texture-format", "png")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4 vertices, 2 triangles", result.stdout)
        data = self.glb.read_bytes()
        self.assertEqual(data[:4], b"glTF")
        self.assertEqual(struct.unpack_from("<I", data, 8)[0], len(data))
        size = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + size])
        self.assertTrue(all("bufferView" in image and "uri" not in image for image in document["images"]))
        # Reload after removing the original bundle: the GLB must stand alone.
        self.obj.unlink()
        (self.root / "quad.mtl").unlink()
        (self.root / "texture.png").unlink()
        scene = trimesh.load_scene(self.glb, process=False)
        self.assertEqual(len(scene.geometry), 1)
        mesh = next(iter(scene.geometry.values()))
        self.assertEqual(len(mesh.vertices), 4)
        self.assertEqual(len(mesh.faces), 2)
        np.testing.assert_allclose(mesh.bounds, [[0, 0, 0], [1, 1, 0]])
        np.testing.assert_allclose(mesh.area, 1)
        self.assertEqual(mesh.visual.uv.shape, (4, 2))
        np.testing.assert_array_equal(np.array(mesh.visual.material.baseColorTexture.convert("RGB")), self.pixels)

    def document_and_images(self):
        data = self.glb.read_bytes()
        size = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20:20 + size])
        binary = data[28 + size:]
        images = []
        for entry in document["images"]:
            view = document["bufferViews"][entry["bufferView"]]
            offset = view.get("byteOffset", 0)
            image = Image.open(BytesIO(binary[offset:offset + view["byteLength"]]))
            image.load()
            images.append(image)
        return document, images

    def assert_reload(self, size):
        scene = trimesh.load_scene(self.glb, process=False)
        mesh = next(iter(scene.geometry.values()))
        self.assertEqual(mesh.visual.material.baseColorTexture.size, size)
        self.assertEqual(mesh.visual.uv.shape, (4, 2))
        self.assertEqual(len(mesh.faces), 2)
        return mesh.visual.material

    def test_texture_limits(self):
        Image.new("RGB", (4096, 4096), (40, 100, 70)).save(self.root / "texture.png")
        for options, side in [((), 2048), (("--max-texture", "512"), 512), (("--max-texture", "0"), 4096)]:
            with self.subTest(side=side):
                result = self.run_converter(None, *options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("4096x4096", result.stdout)
                self.assertIn(f"{side}x{side}", result.stdout)
                self.assertIn("bytes source", result.stdout)
                self.assertIn("bytes WEBP", result.stdout)
                document, images = self.document_and_images()
                self.assertEqual(images[0].size, (side, side))
                self.assertIn("EXT_texture_webp", document["extensionsUsed"])
                self.assertIn("EXT_texture_webp", document["extensionsRequired"])
                self.assert_reload((side, side))

    def test_aspect_ratio_and_lanczos(self):
        pixels = np.random.default_rng(7).integers(0, 256, (60, 120, 3), dtype=np.uint8)
        original = Image.fromarray(pixels)
        original.save(self.root / "texture.png")
        result = self.run_converter(None, "--max-texture", "30", "--texture-format", "png")
        self.assertEqual(result.returncode, 0, result.stderr)
        original.thumbnail((30, 30), Image.Resampling.LANCZOS)
        _, images = self.document_and_images()
        np.testing.assert_array_equal(np.array(images[0]), np.array(original))
        self.assert_reload((30, 15))

    def test_formats_and_quality(self):
        pixels = np.random.default_rng(4).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(self.root / "texture.png")
        for encoding in ("png", "jpeg", "webp"):
            sizes = []
            for quality in (15, 95):
                with self.subTest(encoding=encoding, quality=quality):
                    result = self.run_converter(None, "--texture-format", encoding, "--texture-quality", str(quality))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    document, images = self.document_and_images()
                    self.assertEqual(images[0].format.lower(), encoding)
                    self.assertEqual(document["images"][0]["mimeType"], f"image/{encoding}")
                    texture = document["textures"][0]
                    if encoding == "webp":
                        self.assertNotIn("source", texture)
                        self.assertEqual(texture["extensions"]["EXT_texture_webp"]["source"], 0)
                        for field in ("extensionsUsed", "extensionsRequired"):
                            self.assertIn("EXT_texture_webp", document[field])
                    else:
                        self.assertEqual(texture["source"], 0)
                        self.assertNotIn("EXT_texture_webp", document.get("extensionsUsed", []))
                    sizes.append(self.glb.stat().st_size)
                    self.assert_reload((128, 128))
            if encoding != "png":
                self.assertLess(sizes[0], sizes[1])

    def test_extra_maps_and_displacement(self):
        Image.new("RGB", (64, 32), (128, 128, 255)).save(self.root / "normal.png")
        Image.new("L", (64, 32), 96).save(self.root / "roughness.png")
        Image.new("L", (64, 32), 180).save(self.root / "ao.png")
        # Invalid EXR bytes prove displacement is never even decoded.
        (self.root / "displacement.exr").write_bytes(b"not an image")
        with (self.root / "quad.mtl").open("a") as mtl:
            mtl.write("norm normal.png\nmap_Pr roughness.png\nmap_AO ao.png\ndisp displacement.exr\n")
        for extra in (False, True):
            for encoding in ("png", "jpeg", "webp"):
                with self.subTest(extra=extra, encoding=encoding):
                    options = ["--max-texture", "16", "--texture-format", encoding]
                    if extra:
                        options.append("--extra-maps")
                    result = self.run_converter(None, *options)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    document, images = self.document_and_images()
                    self.assertEqual(len(images), 4 if extra else 1)
                    material = document["materials"][0]
                    pbr = material["pbrMetallicRoughness"]
                    self.assertEqual("normalTexture" in material, extra)
                    self.assertEqual("occlusionTexture" in material, extra)
                    self.assertEqual("metallicRoughnessTexture" in pbr, extra)
                    loaded = self.assert_reload((2, 2))
                    if extra:
                        for image in (loaded.normalTexture, loaded.occlusionTexture, loaded.metallicRoughnessTexture):
                            self.assertEqual(image.size, (16, 8))
                        if encoding == "png":
                            self.assertEqual(loaded.metallicRoughnessTexture.getpixel((0, 0)), (255, 96, 0))
                            self.assertEqual(loaded.occlusionTexture.getpixel((0, 0))[0], 180)

    def test_missing_optional_map_ignored_by_default(self):
        with (self.root / "quad.mtl").open("a") as mtl:
            mtl.write("norm missing.png\ndisp missing.exr\n")
        self.assertEqual(self.run_converter().returncode, 0)
        self.assert_reload((2, 2))
        self.assertNotEqual(self.run_converter(None, "--extra-maps").returncode, 0)

    def test_over_budget_warning(self):
        # A real uncompressed-size PNG export exceeds the real 20 MB threshold.
        pixels = np.random.default_rng(26).integers(0, 256, (2700, 2700, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(self.root / "texture.png")
        result = self.run_converter(None, "--max-texture", "0", "--texture-format", "png")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(self.glb.stat().st_size, 20_000_000)
        warnings = [line for line in result.stdout.splitlines() if line.startswith("Warning:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("20,000,000", warnings[0])
        self.assertIn("--max-texture", warnings[0])
        self.assert_reload((2700, 2700))

    def test_webp_unavailable_falls_back(self):
        output = StringIO()
        with patch("obj_to_glb.features.check", return_value=False), redirect_stdout(output):
            obj_to_glb.convert(self.obj, self.glb)
        self.assertIn("falling back to JPEG", output.getvalue())
        document, images = self.document_and_images()
        self.assertEqual(images[0].format, "JPEG")
        self.assertNotIn("EXT_texture_webp", document.get("extensionsRequired", []))
        self.assert_reload((2, 2))

    def test_bad_webp_declaration_falls_back(self):
        original_export = obj_to_glb.trimesh.exchange.gltf.export_glb

        def broken_export(scene, **kwargs):
            def remove_required(tree):
                tree.pop("extensionsRequired", None)
            return original_export(scene, tree_postprocessor=remove_required, **kwargs)

        output = StringIO()
        with patch("obj_to_glb.trimesh.exchange.gltf.export_glb", side_effect=broken_export), redirect_stdout(output):
            obj_to_glb.convert(self.obj, self.glb)
        self.assertIn("falling back to JPEG", output.getvalue())
        document, images = self.document_and_images()
        self.assertEqual(images[0].format, "JPEG")
        self.assertNotIn("EXT_texture_webp", document.get("extensionsUsed", []))
        self.assert_reload((2, 2))

    def test_alpha_and_small_texture(self):
        pixels = np.array([[[100, 150, 200, 40], [60, 80, 90, 255]]], dtype=np.uint8)
        Image.fromarray(pixels).save(self.root / "texture.png")
        for encoding in ("png", "webp", "jpeg"):
            with self.subTest(encoding=encoding):
                result = self.run_converter(None, "--texture-format", encoding)
                self.assertEqual(result.returncode, 0, result.stderr)
                loaded = self.assert_reload((2, 1))
                if encoding == "jpeg":
                    self.assertIn("JPEG discards texture alpha", result.stdout)
                else:
                    np.testing.assert_array_equal(np.array(loaded.baseColorTexture)[:, :, 3], pixels[:, :, 3])

    def test_invalid_options(self):
        for options in [("--max-texture", "-1"), ("--texture-quality", "101"), ("--texture-quality", "-1")]:
            with self.subTest(options=options):
                self.assertNotEqual(self.run_converter(None, *options).returncode, 0)
                self.assertFalse(self.glb.exists())

    def test_missing_texture_refused(self):
        (self.root / "texture.png").unlink()
        result = self.run_converter()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing texture", result.stderr)
        self.assertFalse(self.glb.exists())

    def test_missing_mtl_refused(self):
        (self.root / "quad.mtl").unlink()
        self.assertNotEqual(self.run_converter().returncode, 0)
        self.assertFalse(self.glb.exists())

    def test_missing_uv_refused(self):
        self.obj.write_text(self.obj.read_text().replace("f 1/1 2/2 3/3 4/4", "f 1 2 3 4"))
        self.assertNotEqual(self.run_converter().returncode, 0)
        self.assertFalse(self.glb.exists())

    def test_obj_output_directory(self):
        self.assertEqual(self.run_converter(self.root).returncode, 0)

    def test_missing_map_refused(self):
        (self.root / "quad.mtl").write_text("newmtl painted\nKd 1 1 1\n")
        self.assertNotEqual(self.run_converter().returncode, 0)
        self.assertFalse(self.glb.exists())


if __name__ == "__main__":
    unittest.main()
