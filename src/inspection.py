"""Headless, one-shot zero-shot inspection built on the existing VisionSystem.

This file deliberately excludes the CustomTkinter GUI, SQLite database, side
camera, homography, and xArm control.  It is the reusable core that will later
be called when UCDavisGUI observes a Bambu print transition to FINISH.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Any

import cv2
import faiss
import numpy as np

from config import DINO_DIM, ROI_SIZE, VISION_TEXT_PROMPT
from inspection_contract import classify_match
from vision import VisionSystem

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ReferenceRecord:
    faiss_id: int
    reference_id: str
    part_id: str
    source_path: str | None


@dataclass(frozen=True)
class InspectionResult:
    schema_version: int
    status: str
    decision: str | None
    reason: str | None
    expected_part: str | None
    detected_part: str | None
    reference_id: str | None
    distance: float | None
    quality_threshold: float
    identity_threshold: float
    prompt: str
    segmentation_found: bool
    centroid: list[int] | None
    inference_ms: float
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ZeroShotInspector:
    """One-shot inspection service with an in-memory FAISS reference index."""

    def __init__(self, vision: VisionSystem | None = None):
        self.vision = vision or VisionSystem()
        self._index = faiss.IndexIDMap2(faiss.IndexFlatL2(DINO_DIM))
        self._records: dict[int, ReferenceRecord] = {}
        self._next_id = 0

    @property
    def reference_count(self) -> int:
        return len(self._records)

    def load_models(self) -> None:
        """Load Grounded-SAM and DINOv2 once."""
        self.vision.load_models()

    @staticmethod
    def _masked_roi(
        frame_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Return the masked 224x224 ROI and object centroid."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr is empty.")
        if mask is None or mask.size == 0:
            raise ValueError("mask is empty.")
        if frame_bgr.shape[:2] != mask.shape[:2]:
            raise ValueError("Mask dimensions do not match the source frame.")

        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            raise ValueError("The segmentation mask has zero area.")

        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        height, width = frame_bgr.shape[:2]
        if width < ROI_SIZE or height < ROI_SIZE:
            raise ValueError(
                f"Frame must be at least {ROI_SIZE}x{ROI_SIZE}; "
                f"received {width}x{height}."
            )

        clean_image = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
        half = ROI_SIZE // 2
        clipped_x = int(np.clip(center_x, half, width - half))
        clipped_y = int(np.clip(center_y, half, height - half))
        x1, y1 = clipped_x - half, clipped_y - half
        roi = clean_image[y1 : y1 + ROI_SIZE, x1 : x1 + ROI_SIZE].copy()

        if roi.shape[:2] != (ROI_SIZE, ROI_SIZE):
            raise RuntimeError("Could not build the fixed-size inspection ROI.")

        return roi, (center_x, center_y)

    def _segment_roi(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame is empty.")
        if not prompt.strip():
            raise ValueError("Grounded-SAM prompt cannot be blank.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = self.vision.segment_with_sam(frame_rgb, prompt.strip())
        if mask is None:
            return None

        roi, centroid = self._masked_roi(frame_bgr, mask)
        return mask, roi, centroid

    def register_reference(
        self,
        *,
        part_id: str,
        frame_bgr: np.ndarray,
        prompt: str = VISION_TEXT_PROMPT,
        reference_id: str | None = None,
        source_path: str | None = None,
    ) -> ReferenceRecord:
        """Segment and add one reference image to the FAISS catalogue."""
        normalized_part = part_id.strip()
        if not normalized_part:
            raise ValueError("part_id cannot be blank.")

        segmented = self._segment_roi(frame_bgr, prompt)
        if segmented is None:
            raise ValueError(
                f"Grounded-SAM found no object in reference image for {part_id!r}."
            )
        _, roi, _ = segmented
        vector = self.vision.extract_embedding(roi)

        faiss_id = self._next_id
        normalized_reference_id = (
            reference_id.strip()
            if reference_id and reference_id.strip()
            else f"{normalized_part}-{faiss_id:04d}"
        )
        record = ReferenceRecord(
            faiss_id=faiss_id,
            reference_id=normalized_reference_id,
            part_id=normalized_part,
            source_path=source_path,
        )

        self._index.add_with_ids(
            vector,
            np.array([faiss_id], dtype=np.int64),
        )
        self._records[faiss_id] = record
        self._next_id += 1
        return record

    def load_reference_directory(
        self,
        root: str | Path,
        *,
        prompt: str = VISION_TEXT_PROMPT,
    ) -> list[ReferenceRecord]:
        """Load references from ``root/<part_id>/<image>`` folders."""
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Reference directory does not exist: {root_path}")

        loaded: list[ReferenceRecord] = []
        candidate_files = sorted(
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not candidate_files:
            raise ValueError(
                "No reference images found. Expected root/<part_id>/*.jpg or *.png."
            )

        for image_path in candidate_files:
            relative = image_path.relative_to(root_path)
            if len(relative.parts) < 2:
                log.warning(
                    "Skipping %s: place reference images inside a part-id folder.",
                    image_path,
                )
                continue

            part_id = relative.parts[0]
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                log.warning("Skipping unreadable reference image: %s", image_path)
                continue

            try:
                loaded.append(
                    self.register_reference(
                        part_id=part_id,
                        frame_bgr=frame,
                        prompt=prompt,
                        reference_id=f"{part_id}:{image_path.stem}",
                        source_path=str(image_path),
                    )
                )
            except ValueError as exc:
                log.warning("Skipping reference %s: %s", image_path, exc)

        if not loaded:
            raise ValueError("No valid references could be registered.")
        return loaded

    def inspect(
        self,
        frame_bgr: np.ndarray,
        *,
        quality_threshold: float,
        identity_threshold: float,
        expected_part: str | None = None,
        prompt: str = VISION_TEXT_PROMPT,
    ) -> tuple[InspectionResult, np.ndarray | None, np.ndarray | None]:
        """Inspect one finished-print frame.

        Returns ``(result, mask, annotated_frame)``.  A missing segmentation is
        returned as ``status=ERROR`` because no quality decision can be made.
        """
        started = time.perf_counter()

        if self.reference_count == 0:
            raise RuntimeError("The reference catalogue is empty.")

        # Validate thresholds before spending time on inference.
        classify_match(
            distance=0.0,
            detected_part="validation",
            expected_part=None,
            quality_threshold=quality_threshold,
            identity_threshold=identity_threshold,
        )

        segmented = self._segment_roi(frame_bgr, prompt)
        if segmented is None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return (
                InspectionResult(
                    schema_version=1,
                    status="ERROR",
                    decision=None,
                    reason=None,
                    expected_part=expected_part,
                    detected_part=None,
                    reference_id=None,
                    distance=None,
                    quality_threshold=float(quality_threshold),
                    identity_threshold=float(identity_threshold),
                    prompt=prompt,
                    segmentation_found=False,
                    centroid=None,
                    inference_ms=round(elapsed_ms, 3),
                    error="Grounded-SAM did not find the requested object.",
                ),
                None,
                frame_bgr.copy(),
            )

        mask, roi, centroid = segmented
        vector = self.vision.extract_embedding(roi)
        distances, indices = self._index.search(vector, 1)
        distance = float(distances[0][0])
        faiss_id = int(indices[0][0])
        record = self._records.get(faiss_id)
        if record is None:
            raise RuntimeError(f"FAISS returned unknown reference id {faiss_id}.")

        match = classify_match(
            distance=distance,
            detected_part=record.part_id,
            expected_part=expected_part,
            quality_threshold=quality_threshold,
            identity_threshold=identity_threshold,
        )

        annotated = frame_bgr.copy()
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(annotated, contours, -1, (255, 255, 255), 2)
        cv2.drawMarker(
            annotated,
            centroid,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        label = f"{match.decision} | {record.part_id} | d={distance:.3f}"
        cv2.putText(
            annotated,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = InspectionResult(
            schema_version=1,
            status="COMPLETE",
            decision=match.decision,
            reason=match.reason,
            expected_part=expected_part,
            detected_part=record.part_id,
            reference_id=record.reference_id,
            distance=distance,
            quality_threshold=float(quality_threshold),
            identity_threshold=float(identity_threshold),
            prompt=prompt,
            segmentation_found=True,
            centroid=[int(centroid[0]), int(centroid[1])],
            inference_ms=round(elapsed_ms, 3),
            error=None,
        )
        return result, mask, annotated
