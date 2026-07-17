import unittest

import numpy as np

from human_mimic_demo.retargeting.o6 import O6Retargeter
from human_mimic_demo.tracking.synthetic import SyntheticTracker


def config():
    return {
        "open_position": [240, 220, 240, 240, 240, 240],
        "closed_position": [40, 50, 40, 40, 40, 40],
        "safe_min": [35, 40, 35, 35, 35, 35],
        "safe_max": [245, 230, 245, 245, 245, 245],
        "low_pass_alpha": 1.0,
        "max_command_step": 255,
    }


class RetargetingTest(unittest.TestCase):
    def test_output_is_six_safe_integer_commands(self):
        retargeter = O6Retargeter(config())
        command, features = retargeter.retarget(SyntheticTracker._hand_landmarks(0.5))
        self.assertEqual(command.shape, (6,))
        self.assertTrue(np.issubdtype(command.dtype, np.integer))
        self.assertTrue(np.all(command >= retargeter.safe_min))
        self.assertTrue(np.all(command <= retargeter.safe_max))
        self.assertTrue(np.all((features.as_array() >= 0.0) & (features.as_array() <= 1.0)))

    def test_more_curl_closes_four_fingers(self):
        retargeter = O6Retargeter(config())
        open_command, _ = retargeter.retarget(SyntheticTracker._hand_landmarks(0.0))
        retargeter.reset_filter()
        closed_command, _ = retargeter.retarget(SyntheticTracker._hand_landmarks(1.0))
        self.assertTrue(np.all(closed_command[[2, 3, 4, 5]] < open_command[[2, 3, 4, 5]]))


if __name__ == "__main__":
    unittest.main()
