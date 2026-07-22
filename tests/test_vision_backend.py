"""Smoke tests for the Windows-friendly Hugging Face model backend."""

from config import GROUNDING_DINO_MODEL_NAME
from vision import VisionSystem


def test_grounding_dino_uses_hugging_face_model() -> None:
    assert GROUNDING_DINO_MODEL_NAME == "IDEA-Research/grounding-dino-tiny"


def test_vision_system_constructs_without_groundingdino_package() -> None:
    vision = VisionSystem()
    assert vision.grounding_processor is None
    assert vision.grounding_dino is None
    assert vision.models_loaded is False
