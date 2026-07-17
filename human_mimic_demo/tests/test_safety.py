import unittest

import numpy as np

from human_mimic_demo.safety import ForceGuard, RelativeArmMapper


class SafetyTest(unittest.TestCase):
    def test_relative_mapping_keeps_orientation_and_clips_workspace(self):
        mapper = RelativeArmMapper.from_config(
            {
                "camera_to_robot_rotation": np.eye(3).tolist(),
                "translation_scale": [1.0, 1.0, 1.0],
                "max_delta_m": [0.2, 0.2, 0.2],
                "max_tcp_step_m": 1.0,
                "workspace_min_m": [0.2, -0.5, 0.1],
                "workspace_max_m": [0.6, 0.2, 0.6],
            }
        )
        origin = np.asarray([0.4, -0.2, 0.3, 0.0, 3.14, 0.0])
        mapper.calibrate(np.asarray([0.0, 0.0, 0.6]), origin)
        target = mapper.target(np.asarray([1.0, 1.0, 1.6]))
        np.testing.assert_allclose(target[:3], [0.6, 0.0, 0.5])
        np.testing.assert_allclose(target[3:], origin[3:])

    def test_force_guard_moves_towards_open(self):
        guard = ForceGuard(
            {"enabled": True, "normal_soft_limit": 100, "retreat_command_step": 5},
            np.asarray([240] * 6),
        )
        command, overridden = guard.apply(
            np.asarray([100] * 6), [[10, 120, 10, 10, 10]]
        )
        self.assertTrue(overridden)
        np.testing.assert_array_equal(command, [105] * 6)


if __name__ == "__main__":
    unittest.main()
