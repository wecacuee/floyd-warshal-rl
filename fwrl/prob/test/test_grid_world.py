
from unittest import TestCase
from pkg_resources import resource_stream

import numpy as np
from PIL import Image

from umcog import draw

from ..windy_grid_world import WindyGridWorld

class TestGridWorld(TestCase):
    def setUp(self):
        self.lgw = WindyGridWorld.from_maze_name(maze_name = "4-room-lava-world.txt")

    def test_maze_name(self):
        self.lgw = WindyGridWorld.from_maze_name(maze_name = "4-room-lava-world.txt")

    def test_free_space(self):
        next_pose = np.array([1, 3])
        p, r, d = self.lgw.step(np.array([1, 2]), next_pose)
        self.assertTrue(np.all(p == next_pose))
        self.assertEqual(r, 0)
        self.assertEqual(d, False)

    def test_wall(self):
        pose = np.array([1, 2])
        next_pose = np.array([0, 2])
        p, r, d = self.lgw.step(pose, next_pose)
        self.assertTrue(np.all(p == pose))
        self.assertEqual(r, -0.05)
        self.assertEqual(d, False)

    def test_lava(self):
        pose = np.array([1, 2])
        next_pose = np.array([2, 2])
        p, r, d = self.lgw.step(pose, next_pose)
        self.assertTrue(np.all(p == np.array(None)))
        self.assertEqual(r, -10)
        self.assertEqual(d, True)

    def test_render(self, expected_img_rsrc = "data/4-room-lava-world.png"):
        cnvs = self.lgw.render(None, 100)
        expected_img = np.asarray(
            Image.open(resource_stream(__name__, expected_img_rsrc)))
        expected_img = expected_img[:, :, :3]
        img_argb = draw.to_ndarray(cnvs)
        img = img_argb[:, :, 1:].copy()
        #draw.imwrite("/tmp/test.png", cnvs)
        #print("exp:", expected_img.shape, expected_img[:5, :5, 0])
        #Image.fromarray(expected_img).save("/tmp/exp.png")
        #print("got:", img.shape, img[:5, :5, 0])
        #Image.fromarray(img).save("/tmp/got.png")
        #diff = (expected_img == img).view('u1')*255
        #print(diff.shape)
        #Image.fromarray(diff).save("/tmp/diff.png")
        self.assertTrue(np.all(img == expected_img))

    def test_valid_random_pos(self):
        rpos = self.lgw.valid_random_pos()
        rcell = self.lgw.cell_code(rpos)
        self.assertTrue(rcell in ([self.lgw.CELL_FREE] + self.lgw.CELL_WIND_NEWS))
