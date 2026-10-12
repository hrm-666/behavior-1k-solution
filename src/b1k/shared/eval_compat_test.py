import unittest

import numpy as np

from b1k.shared.eval_compat import CAMERA_KEYS
from b1k.shared.eval_compat import NEW_PROPRIO_DIM
from b1k.shared.eval_compat import NEW_PROPRIO_INDICES
from b1k.shared.eval_compat import OLD_PROPRIO_DIM
from b1k.shared.eval_compat import OLD_PROPRIO_INDICES
from b1k.shared.eval_compat import PROPRIO_KEY
from b1k.shared.eval_compat import adapt_single_env_observation
from b1k.shared.eval_compat import extract_state_from_proprio


def _make_proprio(dim: int, indices: dict[str, slice]) -> tuple[np.ndarray, np.ndarray]:
    proprio = np.zeros(dim, dtype=np.float32)
    proprio[indices["base_qvel"]] = [1, 2, 3]
    proprio[indices["trunk_qpos"]] = [4, 5, 6, 7]
    proprio[indices["arm_left_qpos"]] = np.arange(8, 15)
    proprio[indices["gripper_left_qpos"]] = [0.025, 0.025]
    proprio[indices["arm_right_qpos"]] = np.arange(15, 22)
    proprio[indices["gripper_right_qpos"]] = [0.05, 0.05]

    expected = np.concatenate(
        [
            np.arange(1, 15, dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.arange(15, 22, dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ]
    )
    return proprio, expected


class ExtractStateFromProprioTest(unittest.TestCase):
    def test_current_61_dim_layout(self):
        proprio, expected = _make_proprio(NEW_PROPRIO_DIM, NEW_PROPRIO_INDICES)

        state = extract_state_from_proprio(proprio)

        np.testing.assert_allclose(state, expected)
        self.assertEqual(state.shape, (23,))

    def test_2025_256_dim_layout(self):
        proprio, expected = _make_proprio(OLD_PROPRIO_DIM, OLD_PROPRIO_INDICES)

        state = extract_state_from_proprio(proprio)

        np.testing.assert_allclose(state, expected)
        self.assertEqual(state.shape, (23,))

    def test_preserves_leading_dimensions(self):
        proprio, expected = _make_proprio(NEW_PROPRIO_DIM, NEW_PROPRIO_INDICES)

        state = extract_state_from_proprio(np.stack([proprio, proprio]))

        np.testing.assert_allclose(state, np.stack([expected, expected]))
        self.assertEqual(state.shape, (2, 23))

    def test_rejects_unknown_layout(self):
        with self.assertRaisesRegex(ValueError, "Unsupported R1Pro proprioception dimension 62"):
            extract_state_from_proprio(np.zeros(62, dtype=np.float32))


class AdaptSingleEnvObservationTest(unittest.TestCase):
    def test_canonicalizes_emerge_robot_namespace(self):
        obs = {
            "robot::proprio": np.zeros((1, NEW_PROPRIO_DIM), dtype=np.float32),
            "robot::robot:zed_link:Camera:0::rgb": np.zeros((1, 12, 10, 4), dtype=np.uint8),
            "robot::robot:left_realsense_link:Camera:0::rgb": np.zeros((1, 8, 6, 4), dtype=np.uint8),
            "robot::robot:right_realsense_link:Camera:0::rgb": np.zeros((1, 8, 6, 4), dtype=np.uint8),
        }

        adapted = adapt_single_env_observation(obs)

        self.assertEqual(adapted[PROPRIO_KEY].shape, (NEW_PROPRIO_DIM,))
        self.assertEqual(adapted[CAMERA_KEYS[0]].shape, (12, 10, 4))

    def test_removes_current_evaluator_batch(self):
        obs = {
            PROPRIO_KEY: np.zeros((1, NEW_PROPRIO_DIM), dtype=np.float32),
            CAMERA_KEYS[0]: np.zeros((1, 224, 224, 4), dtype=np.uint8),
            CAMERA_KEYS[1]: np.zeros((1, 224, 224, 4), dtype=np.uint8),
            CAMERA_KEYS[2]: np.zeros((1, 224, 224, 4), dtype=np.uint8),
            "task_id": np.array([[0]], dtype=np.int64),
        }

        adapted = adapt_single_env_observation(obs)

        self.assertEqual(adapted[PROPRIO_KEY].shape, (NEW_PROPRIO_DIM,))
        self.assertEqual(adapted[CAMERA_KEYS[0]].shape, (224, 224, 4))
        self.assertEqual(adapted[CAMERA_KEYS[1]].shape, (224, 224, 4))
        self.assertEqual(adapted[CAMERA_KEYS[2]].shape, (224, 224, 4))
        self.assertEqual(adapted["task_id"].shape, (1,))
        self.assertEqual(obs[PROPRIO_KEY].shape, (1, NEW_PROPRIO_DIM))

    def test_preserves_2025_unbatched_observation(self):
        obs = {
            PROPRIO_KEY: np.zeros(OLD_PROPRIO_DIM, dtype=np.float32),
            CAMERA_KEYS[0]: np.zeros((224, 224, 4), dtype=np.uint8),
            CAMERA_KEYS[1]: np.zeros((224, 224, 4), dtype=np.uint8),
            CAMERA_KEYS[2]: np.zeros((224, 224, 4), dtype=np.uint8),
            "task_id": np.array([0], dtype=np.int64),
        }

        adapted = adapt_single_env_observation(obs)

        self.assertEqual(adapted[PROPRIO_KEY].shape, (OLD_PROPRIO_DIM,))
        self.assertEqual(adapted[CAMERA_KEYS[0]].shape, (224, 224, 4))
        self.assertEqual(adapted["task_id"].shape, (1,))

    def test_rejects_multiple_environments(self):
        obs = {
            PROPRIO_KEY: np.zeros((2, NEW_PROPRIO_DIM), dtype=np.float32),
            CAMERA_KEYS[0]: np.zeros((2, 12, 10, 4), dtype=np.uint8),
            CAMERA_KEYS[1]: np.zeros((2, 8, 6, 4), dtype=np.uint8),
            CAMERA_KEYS[2]: np.zeros((2, 8, 6, 4), dtype=np.uint8),
        }
        with self.assertRaisesRegex(ValueError, "--num-envs 1"):
            adapt_single_env_observation(obs)
if __name__ == "__main__":
    unittest.main()
