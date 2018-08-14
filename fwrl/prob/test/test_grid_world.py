
from doctest import DocTestSuite
from unittest import TestCase
from pkg_resources import resource_stream

import numpy as np
from PIL import Image

from umcog import draw

from .. import windy_grid_world
from ..windy_grid_world import WindyGridWorld, shortest_path, Act2DSpace

def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(windy_grid_world))
    return tests

class TestGridWorld(TestCase):
    def setUp(self):
        self.free_space_reward = 0
        self.wall_reward = -0.05
        self.lava_reward = -10
        self.lgw = WindyGridWorld.from_maze_name(
            maze_name = "4-room-lava-world",
            free_space_reward = self.free_space_reward,
            wall_reward = self.wall_reward,
            lava_reward = self.lava_reward)

    def test_free_space(self):
        next_pose = np.array([1, 3])
        p, r, d = self.lgw.step(np.array([1, 2]), next_pose)
        self.assertTrue(np.all(p == next_pose))
        self.assertEqual(r, self.free_space_reward)
        self.assertEqual(d, False)

    def test_wall(self):
        pose = np.array([1, 2])
        next_pose = np.array([0, 2])
        p, r, d = self.lgw.step(pose, next_pose)
        self.assertTrue(np.all(p == pose))
        self.assertEqual(r, self.wall_reward)
        self.assertEqual(d, False)

    def test_lava(self):
        pose = np.array([1, 2])
        next_pose = np.array([2, 2])
        p, r, d = self.lgw.step(pose, next_pose)
        self.assertTrue(np.all(p == np.array(None)))
        self.assertEqual(r, self.lava_reward)
        self.assertEqual(d, True)

    def test_render(self, expected_img_rsrc = "data/4-room-lava-world.png"):
        cnvs = self.lgw.render(None, 100)
        expected_img = np.asarray(
            Image.open(resource_stream(__name__, expected_img_rsrc)))
        expected_img = expected_img[:, :, :3]
        img_argb = draw.to_ndarray(cnvs)
        img = img_argb[:, :, 1:].copy()
        draw.imwrite("/tmp/4-room-lava-world.png", cnvs)
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

    def test_shortest_path(self):
        for _ in range(5):
            start = self.lgw.valid_random_pos()
            end = self.lgw.valid_random_pos()
            action_space = Act2DSpace(np.random.RandomState(0))
            length, path = shortest_path(self.lgw, start, end, action_space)
            rev_length, rev_path = shortest_path(self.lgw, end, start, action_space)
            self.assertEqual(length, rev_length)
            self.assertTrue(all(s1 == s2 for s1, s2 in zip(path, reversed(rev_path))))

