"""CPU-only conversion tests; never run a photogrammetry session."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

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

    def run_converter(self, source=None):
        return subprocess.run(
            [sys.executable, str(CONVERTER), str(source or self.obj), str(self.glb)],
            capture_output=True, text=True, check=False,
        )

    def test_textured_quad_round_trip(self):
        result = self.run_converter()
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
