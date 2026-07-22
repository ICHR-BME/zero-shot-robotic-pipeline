import pytest

from inspection_contract import classify_match


def test_approved_inside_quality_threshold():
    result = classify_match(
        distance=20.0,
        detected_part="gear-v3",
        expected_part="gear-v3",
        quality_threshold=40.0,
        identity_threshold=120.0,
    )
    assert result.decision == "APPROVED"
    assert result.reason == "WITHIN_QUALITY_THRESHOLD"


def test_rejected_between_quality_and_identity_thresholds():
    result = classify_match(
        distance=60.0,
        detected_part="gear-v3",
        expected_part="gear-v3",
        quality_threshold=40.0,
        identity_threshold=120.0,
    )
    assert result.decision == "REJECTED"
    assert result.reason == "QUALITY_DISTANCE_EXCEEDED"


def test_unknown_above_identity_threshold():
    result = classify_match(
        distance=121.0,
        detected_part="gear-v3",
        expected_part="gear-v3",
        quality_threshold=40.0,
        identity_threshold=120.0,
    )
    assert result.decision == "UNKNOWN"
    assert result.reason == "IDENTITY_DISTANCE_EXCEEDED"


def test_recognized_wrong_part_is_rejected():
    result = classify_match(
        distance=10.0,
        detected_part="bracket-v2",
        expected_part="gear-v3",
        quality_threshold=40.0,
        identity_threshold=120.0,
    )
    assert result.decision == "REJECTED"
    assert result.reason == "WRONG_PART"


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(ValueError, match="quality_threshold"):
        classify_match(
            distance=10.0,
            detected_part="gear-v3",
            quality_threshold=130.0,
            identity_threshold=120.0,
        )
