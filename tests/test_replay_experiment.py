import unittest

import cv2
import numpy as np

from experiments.replay_experiment import projective_center, pose_errors


class ReplayExperimentTests(unittest.TestCase):
    def test_center_recovers_projected_tag_origin_under_perspective(self):
        local = np.array([[-.05, .05, 0], [.05, .05, 0], [.05, -.05, 0], [-.05, -.05, 0], [0, 0, 0.]])
        K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
        rv, tv = np.array([.6, -.4, .2]), np.array([.02, -.03, .4])
        points = cv2.projectPoints(local, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
        np.testing.assert_allclose(projective_center(points[:4]), points[4], atol=1e-10)
        self.assertGreater(np.linalg.norm(points[:4].mean(0) - points[4]), .1)

    def test_errors_use_millimetres_and_so3_angle(self):
        truth = np.eye(4)
        truth[:3, 3] = [100, 200, 300]
        pose = np.eye(4)
        pose[:3, 3] = [.103, .204, .3]
        pose[:3, :3] = cv2.Rodrigues(np.array([0., 0., np.pi / 6]))[0]
        errors = pose_errors(pose, truth)
        self.assertAlmostEqual(errors['position_error_mm'], 5.)
        self.assertAlmostEqual(errors['rotation_error_deg'], 30.)

    def test_degenerate_tag_center_is_rejected(self):
        with self.assertRaises(ValueError):
            projective_center(np.zeros((4, 2)))


if __name__ == '__main__':
    unittest.main()
