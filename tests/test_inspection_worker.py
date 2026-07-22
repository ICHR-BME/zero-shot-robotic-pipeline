from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from config import DEFAULT_IDENTITY_TOLERANCE, DEFAULT_QUALITY_TOLERANCE
from inspection_worker import InspectionWorker


@dataclass
class FakeResult:
    status: str = "COMPLETE"
    decision: str = "APPROVED"

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "status": self.status,
            "decision": self.decision,
            "reason": "WITHIN_QUALITY_THRESHOLD",
            "expected_part": "test-part",
            "detected_part": "test-part",
            "reference_id": "test-part:reference-01",
            "distance": 500.0,
            "quality_threshold": 750.0,
            "identity_threshold": 1000.0,
            "prompt": "plastic object",
            "segmentation_found": True,
            "centroid": [10, 10],
            "inference_ms": 1.0,
            "error": None,
        }


class FakeInspector:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def inspect(self, frame_bgr, **kwargs):
        self.calls.append(kwargs)
        mask = np.full(frame_bgr.shape[:2], 255, dtype=np.uint8)
        return FakeResult(), mask, frame_bgr.copy()


def _write_job(root: Path, job_id: str, payload: dict | None = None) -> Path:
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(job_dir / "source.jpg"), image)
    request = {
        "schema_version": 1,
        "inspection_id": job_id,
        "source_image": "source.jpg",
        "expected_part": "test-part",
        "prompt": "plastic object",
    }
    if payload:
        request.update(payload)
    (job_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    return job_dir


def test_worker_processes_job_with_config_defaults(tmp_path: Path) -> None:
    inspector = FakeInspector()
    job_dir = _write_job(tmp_path, "job-001")
    worker = InspectionWorker(jobs_dir=tmp_path, inspector=inspector)

    assert worker.run_pending() == 1
    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))

    assert result["status"] == "COMPLETE"
    assert result["decision"] == "APPROVED"
    assert result["inspection_id"] == "job-001"
    assert result["quality_threshold"] == float(DEFAULT_QUALITY_TOLERANCE)
    assert result["identity_threshold"] == float(DEFAULT_IDENTITY_TOLERANCE)
    assert result["mask_image"] == "mask.png"
    assert result["annotated_image"] == "annotated.jpg"
    assert (job_dir / "mask.png").is_file()
    assert (job_dir / "annotated.jpg").is_file()
    assert not (job_dir / ".worker.lock").exists()
    assert inspector.calls[0]["quality_threshold"] == float(
        DEFAULT_QUALITY_TOLERANCE
    )
    assert inspector.calls[0]["identity_threshold"] == float(
        DEFAULT_IDENTITY_TOLERANCE
    )


def test_completed_job_is_not_processed_twice(tmp_path: Path) -> None:
    inspector = FakeInspector()
    job_dir = _write_job(tmp_path, "job-002")
    worker = InspectionWorker(jobs_dir=tmp_path, inspector=inspector)

    assert worker.run_pending() == 1
    first_result = (job_dir / "result.json").read_bytes()
    assert worker.run_pending() == 0
    assert (job_dir / "result.json").read_bytes() == first_result
    assert len(inspector.calls) == 1


def test_path_traversal_becomes_error_result(tmp_path: Path) -> None:
    inspector = FakeInspector()
    job_dir = _write_job(
        tmp_path,
        "job-003",
        {"source_image": "../outside.jpg"},
    )
    worker = InspectionWorker(jobs_dir=tmp_path, inspector=inspector)

    assert worker.run_pending() == 1
    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "ERROR"
    assert result["reason"] == "WORKER_ERROR"
    assert "cannot leave the job directory" in result["error"]
    assert inspector.calls == []


def test_invalid_threshold_order_becomes_error_result(tmp_path: Path) -> None:
    inspector = FakeInspector()
    job_dir = _write_job(
        tmp_path,
        "job-004",
        {"quality_threshold": 1001, "identity_threshold": 1000},
    )
    worker = InspectionWorker(jobs_dir=tmp_path, inspector=inspector)

    assert worker.run_pending() == 1
    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "ERROR"
    assert "quality_threshold must be less than or equal" in result["error"]
    assert inspector.calls == []


def test_stale_lock_is_reclaimed(tmp_path: Path) -> None:
    inspector = FakeInspector()
    job_dir = _write_job(tmp_path, "job-005")
    lock_path = job_dir / ".worker.lock"
    lock_path.write_text("old", encoding="utf-8")
    old = time.time() - 30
    os.utime(lock_path, (old, old))

    worker = InspectionWorker(
        jobs_dir=tmp_path,
        inspector=inspector,
        stale_lock_s=1,
    )
    assert worker.run_pending() == 1
    assert (job_dir / "result.json").is_file()
    assert not lock_path.exists()
