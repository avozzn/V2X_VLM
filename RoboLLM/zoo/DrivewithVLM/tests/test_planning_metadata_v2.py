import numpy as np
import pytest

from projects.mmdet3d_plugin.datasets.data_utils.planning_metadata_v2 import (
    generate_planning_metadata_v2,
)
from tools.eval.trajectory_diagnostics_v2 import parse_answer


MASK = np.ones((1, 9, 2), dtype=np.float32)


def metadata(points, velocity):
    return generate_planning_metadata_v2(
        sdc_planning=np.asarray(points, dtype=np.float64)[None, ...],
        planning_mask=MASK,
        current_velocity=np.asarray(velocity, dtype=np.float64),
        dt=0.5,
    )


def test_keep_lane_keep_speed():
    result = metadata([(2.0 * index, 0.0) for index in range(1, 10)], [4.0, 0.0])
    assert result["lateral_action"] == "KEEP_LANE"
    assert result["longitudinal_action"] == "KEEP_SPEED"


def test_lane_change_and_turn_are_separated_by_final_heading():
    lane_change = metadata([(2.0 * index, 0.4 * index) for index in range(1, 10)], [4.0, 0.0])
    turn = metadata([(1.5 * index, 0.2 * index * index) for index in range(1, 10)], [3.0, 0.0])
    assert lane_change["lateral_action"] == "LANE_CHANGE_LEFT"
    assert turn["lateral_action"] == "TURN_LEFT"


def test_longitudinal_accelerate_decelerate_and_stop():
    accelerating_x = np.cumsum(np.linspace(0.5, 4.5, 9))
    decelerating_x = np.cumsum(np.linspace(4.5, 0.5, 9))
    stopped_x = np.cumsum(np.full(9, 0.02))
    assert metadata([(x, 0.0) for x in accelerating_x], [1.0, 0.0])["longitudinal_action"] == "ACCELERATE"
    assert metadata([(x, 0.0) for x in decelerating_x], [9.0, 0.0])["longitudinal_action"] == "DECELERATE"
    assert metadata([(x, 0.0) for x in stopped_x], [0.0, 0.0])["longitudinal_action"] == "STOP"


def test_mask_cannot_be_passed_as_velocity():
    points = np.asarray([(2.0 * index, 0.0) for index in range(1, 10)])
    with pytest.raises((TypeError, ValueError), match="current_velocity"):
        generate_planning_metadata_v2(
            sdc_planning=points[None, ...],
            planning_mask=MASK,
            current_velocity=MASK,
        )


def test_v2_answer_parser_rejects_legacy_and_accepts_action_trajectory():
    legacy = "Meta Action:MOVE FORWARD WITH CONSTANT SPEED\nTrajectory:\n[(1,0)]"
    with pytest.raises(ValueError, match="lateral_action"):
        parse_answer(legacy)
    points = ", ".join(f"({index:.2f},0.00)" for index in range(1, 10))
    answer = (
        "Lateral Action: KEEP_LANE\n"
        "Longitudinal Action: KEEP_SPEED\n"
        f"Trajectory:\n[{points}]"
    )
    parsed = parse_answer(answer)
    assert parsed["trajectory"].shape == (9, 2)


def test_continuous_values_are_derived_from_trajectory():
    result = metadata([(2.0 * index, 0.0) for index in range(1, 10)], [4.0, 0.0])
    assert result["target_end_speed_mps"] == pytest.approx(4.0)
    assert result["mean_acceleration_mps2"] == pytest.approx(0.0, abs=1e-12)


def test_answer_requires_exactly_nine_finite_points():
    with pytest.raises(ValueError, match="trajectory must be finite"):
        parse_answer(
            "Lateral Action: KEEP_LANE\nLongitudinal Action: KEEP_SPEED\n"
            "Trajectory:\n[(1.0,0.0)]")
