"""Compatibility helpers for connecting 2025 B1K policies to current eval."""

import numpy as np


OLD_PROPRIO_DIM = 256
NEW_PROPRIO_DIM = 61

# The 2025 evaluator emitted OmniGibson's full R1Pro proprioception vector. The
# current evaluator emits only the fields selected by omnigibson/eval/r1pro.yaml.
# Keep both layouts local to the policy adapter so the inference server does not
# accidentally interpret a vector using indices from whichever OmniGibson
# version happens to be installed in its environment.
OLD_PROPRIO_INDICES = {
    "base_qvel": np.s_[253:256],
    "trunk_qpos": np.s_[236:240],
    "arm_left_qpos": np.s_[158:165],
    "gripper_left_qpos": np.s_[193:195],
    "arm_right_qpos": np.s_[197:204],
    "gripper_right_qpos": np.s_[232:234],
}

NEW_PROPRIO_INDICES = {
    "base_qvel": np.s_[0:3],
    "arm_left_qpos": np.s_[3:10],
    "gripper_left_qpos": np.s_[24:26],
    "arm_right_qpos": np.s_[28:35],
    "gripper_right_qpos": np.s_[49:51],
    "trunk_qpos": np.s_[53:57],
}

PROPRIO_KEY = "robot_r1::proprio"
CAMERA_KEYS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)

# The official evaluator config names the robot ``robot_r1``. EMERGE's config
# names the same robot ``robot`` so that its observation gateway receives the
# namespace it already understands. Canonicalize both wire formats to the
# original checkpoint keys before the champion wrapper sees them.
PROPRIO_KEY_ALIASES = (
    PROPRIO_KEY,
    "robot::proprio",
    "robot::robot:proprio",
)
CAMERA_KEY_ALIASES = (
    (
        CAMERA_KEYS[0],
        "robot::robot:zed_link:Camera:0::rgb",
    ),
    (
        CAMERA_KEYS[1],
        "robot::robot:left_realsense_link:Camera:0::rgb",
    ),
    (
        CAMERA_KEYS[2],
        "robot::robot:right_realsense_link:Camera:0::rgb",
    ),
)


def canonicalize_observation_keys(obs: dict) -> dict:
    """Return an observation containing the champion checkpoint's key names."""

    adapted = dict(obs)
    key_groups = ((PROPRIO_KEY, PROPRIO_KEY_ALIASES), *zip(CAMERA_KEYS, CAMERA_KEY_ALIASES, strict=True))
    for canonical_key, aliases in key_groups:
        source_key = next((key for key in aliases if key in obs), None)
        if source_key is None:
            available = ", ".join(sorted(map(str, obs.keys())))
            raise KeyError(
                f"Missing required observation {canonical_key!r}; "
                f"accepted aliases={aliases!r}; available keys=[{available}]"
            )
        adapted[canonical_key] = obs[source_key]
    return adapted


def extract_state_from_proprio(proprio_data):
    """Extract the 23-D model state from either supported R1Pro layout.

    The two gripper fingers are assumed to be perfectly correlated, matching
    the representation used to train the checkpoint.
    """
    proprio_data = np.asarray(proprio_data)
    if proprio_data.ndim == 0:
        raise ValueError("R1Pro proprioception must have at least one dimension")

    proprio_dim = proprio_data.shape[-1]
    if proprio_dim == OLD_PROPRIO_DIM:
        indices = OLD_PROPRIO_INDICES
    elif proprio_dim == NEW_PROPRIO_DIM:
        indices = NEW_PROPRIO_INDICES
    else:
        raise ValueError(
            f"Unsupported R1Pro proprioception dimension {proprio_dim}; "
            f"expected {NEW_PROPRIO_DIM} (current evaluator) or {OLD_PROPRIO_DIM} (2025 evaluator)"
        )

    base_qvel = proprio_data[..., indices["base_qvel"]]
    trunk_qpos = proprio_data[..., indices["trunk_qpos"]]
    arm_left_qpos = proprio_data[..., indices["arm_left_qpos"]]
    arm_right_qpos = proprio_data[..., indices["arm_right_qpos"]]

    left_gripper_raw = proprio_data[..., indices["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., indices["gripper_right_qpos"]].sum(axis=-1, keepdims=True)

    max_gripper_width = 0.1
    left_gripper_width = 2.0 * (left_gripper_raw / max_gripper_width) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / max_gripper_width) - 1.0

    state = np.concatenate(
        [
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            left_gripper_width,
            arm_right_qpos,
            right_gripper_width,
        ],
        axis=-1,
    )
    if state.shape[-1] != 23:
        raise ValueError(f"Expected a 23-D policy state, got shape {state.shape}")
    return state


def adapt_single_env_observation(obs: dict) -> dict:
    """Canonicalize and unbatch one current- or legacy-evaluator observation."""
    adapted = canonicalize_observation_keys(obs)

    proprio = np.asarray(adapted[PROPRIO_KEY])
    if proprio.ndim == 2:
        if proprio.shape[0] != 1:
            raise ValueError(
                f"B1KPolicyWrapper supports exactly one environment, got proprio shape {proprio.shape}. "
                "Run evaluation with --num-envs 1."
            )
        proprio = proprio[0]
    adapted[PROPRIO_KEY] = proprio

    for key in CAMERA_KEYS:
        image = np.asarray(adapted[key])
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    f"B1KPolicyWrapper supports exactly one environment, got {key} shape {image.shape}. "
                    "Run evaluation with --num-envs 1."
                )
            image = image[0]
        adapted[key] = image

    if "task_id" in adapted:
        task_id = np.asarray(adapted["task_id"])
        if task_id.ndim == 2:
            if task_id.shape[0] != 1:
                raise ValueError(
                    f"B1KPolicyWrapper supports exactly one environment, got task_id shape {task_id.shape}. "
                    "Run evaluation with --num-envs 1."
                )
            task_id = task_id[0]
        adapted["task_id"] = task_id

    return adapted
