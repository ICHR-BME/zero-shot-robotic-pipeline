"""Headless file-based worker for finished-print zero-shot inspection.

The worker watches a jobs directory. Each job is a folder containing a
``request.json`` file and the source image named by that request. Results are
written back into the same folder using atomic file replacement.

This module intentionally contains no printer, Tulip, GUI, robot, or SQLite
logic. It is the stable boundary that UCDavisGUI will call later.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Protocol

import cv2
import numpy as np

from config import (
    DEFAULT_IDENTITY_TOLERANCE,
    DEFAULT_QUALITY_TOLERANCE,
    VISION_TEXT_PROMPT,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
REQUEST_FILE = "request.json"
RESULT_FILE = "result.json"
LOCK_FILE = ".worker.lock"
DEFAULT_STALE_LOCK_S = 600.0


class InspectorProtocol(Protocol):
    """The small interface required from ``ZeroShotInspector``."""

    def inspect(
        self,
        frame_bgr: np.ndarray,
        *,
        quality_threshold: float,
        identity_threshold: float,
        expected_part: str | None = None,
        prompt: str = VISION_TEXT_PROMPT,
    ) -> tuple[Any, np.ndarray | None, np.ndarray | None]: ...


class RequestError(ValueError):
    """Raised when a worker request is malformed or unsafe."""


@dataclass(frozen=True)
class InspectionRequest:
    schema_version: int
    inspection_id: str
    source_image: str
    expected_part: str | None
    prompt: str
    quality_threshold: float
    identity_threshold: float

    @classmethod
    def from_dict(cls, payload: Any, *, job_dir: Path) -> "InspectionRequest":
        if not isinstance(payload, dict):
            raise RequestError("request.json must contain a JSON object.")

        schema_version = payload.get("schema_version", SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise RequestError(
                f"Unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION}."
            )

        inspection_id = payload.get("inspection_id", job_dir.name)
        if not isinstance(inspection_id, str) or not inspection_id.strip():
            raise RequestError("inspection_id must be a non-empty string.")
        inspection_id = inspection_id.strip()
        if inspection_id != job_dir.name:
            raise RequestError(
                "inspection_id must exactly match the name of its job directory."
            )

        source_image = payload.get("source_image", "source.jpg")
        if not isinstance(source_image, str) or not source_image.strip():
            raise RequestError("source_image must be a non-empty relative path.")
        source_image = source_image.strip()
        _resolve_inside(job_dir, source_image)

        expected_part = payload.get("expected_part")
        if expected_part is not None:
            if not isinstance(expected_part, str):
                raise RequestError("expected_part must be a string or null.")
            expected_part = expected_part.strip() or None

        prompt = payload.get("prompt", VISION_TEXT_PROMPT)
        if not isinstance(prompt, str) or not prompt.strip():
            raise RequestError("prompt must be a non-empty string.")
        prompt = prompt.strip()

        quality_threshold = _finite_float(
            payload.get("quality_threshold", DEFAULT_QUALITY_TOLERANCE),
            "quality_threshold",
        )
        identity_threshold = _finite_float(
            payload.get("identity_threshold", DEFAULT_IDENTITY_TOLERANCE),
            "identity_threshold",
        )
        if quality_threshold < 0 or identity_threshold < 0:
            raise RequestError("Thresholds must be non-negative.")
        if quality_threshold > identity_threshold:
            raise RequestError(
                "quality_threshold must be less than or equal to identity_threshold."
            )

        return cls(
            schema_version=SCHEMA_VERSION,
            inspection_id=inspection_id,
            source_image=source_image,
            expected_part=expected_part,
            prompt=prompt,
            quality_threshold=quality_threshold,
            identity_threshold=identity_threshold,
        )


def _finite_float(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{field} must be a number.") from exc
    if not math.isfinite(converted):
        raise RequestError(f"{field} must be finite.")
    return converted


def _resolve_inside(job_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise RequestError("source_image must be relative to the job directory.")

    root = job_dir.resolve()
    candidate = (job_dir / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RequestError("source_image cannot leave the job directory.") from exc
    return candidate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


def _atomic_write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported output image extension: {suffix}")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"OpenCV could not encode {path.name}.")
    _atomic_write_bytes(path, encoded.tobytes())


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


class InspectionWorker:
    """Process pending job directories with one already-loaded inspector."""

    def __init__(
        self,
        *,
        jobs_dir: str | Path,
        inspector: InspectorProtocol,
        poll_interval_s: float = 0.5,
        stale_lock_s: float = DEFAULT_STALE_LOCK_S,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be greater than zero.")
        if stale_lock_s <= 0:
            raise ValueError("stale_lock_s must be greater than zero.")
        self.jobs_dir = Path(jobs_dir)
        self.inspector = inspector
        self.poll_interval_s = float(poll_interval_s)
        self.stale_lock_s = float(stale_lock_s)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def pending_job_dirs(self) -> list[Path]:
        return sorted(
            path
            for path in self.jobs_dir.iterdir()
            if path.is_dir()
            and (path / REQUEST_FILE).is_file()
            and not (path / RESULT_FILE).exists()
        )

    def run_pending(self) -> int:
        processed = 0
        for job_dir in self.pending_job_dirs():
            if self.process_job(job_dir):
                processed += 1
        return processed

    def serve_forever(self) -> None:
        log.info("Inspection worker watching %s", self.jobs_dir.resolve())
        while True:
            self.run_pending()
            time.sleep(self.poll_interval_s)

    def process_job(self, job_dir: str | Path) -> bool:
        job_path = Path(job_dir)
        if (job_path / RESULT_FILE).exists():
            return False
        if not self._acquire_lock(job_path):
            return False

        request: InspectionRequest | None = None
        try:
            request_payload = _read_json(job_path / REQUEST_FILE)
            request = InspectionRequest.from_dict(request_payload, job_dir=job_path)
            source_path = _resolve_inside(job_path, request.source_image)
            frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RequestError(f"Could not read source image: {request.source_image}")

            result, mask, annotated = self.inspector.inspect(
                frame,
                expected_part=request.expected_part,
                quality_threshold=request.quality_threshold,
                identity_threshold=request.identity_threshold,
                prompt=request.prompt,
            )

            mask_name: str | None = None
            annotated_name: str | None = None
            if mask is not None:
                mask_name = "mask.png"
                _atomic_write_image(job_path / mask_name, mask)
            if annotated is not None:
                annotated_name = "annotated.jpg"
                _atomic_write_image(job_path / annotated_name, annotated)

            if not hasattr(result, "to_dict"):
                raise TypeError("Inspector result must provide to_dict().")
            payload = dict(result.to_dict())
            payload.update(
                {
                    "inspection_id": request.inspection_id,
                    "source_image": request.source_image,
                    "mask_image": mask_name,
                    "annotated_image": annotated_name,
                    "completed_at": _utc_now(),
                }
            )
            _atomic_write_json(job_path / RESULT_FILE, payload)
            log.info(
                "Inspection %s completed: %s",
                request.inspection_id,
                payload.get("decision") or payload.get("status"),
            )
            return True
        except Exception as exc:  # The worker must record failures, not crash-loop.
            inspection_id = request.inspection_id if request else job_path.name
            error_payload = {
                "schema_version": SCHEMA_VERSION,
                "inspection_id": inspection_id,
                "status": "ERROR",
                "decision": None,
                "reason": "WORKER_ERROR",
                "expected_part": request.expected_part if request else None,
                "detected_part": None,
                "reference_id": None,
                "distance": None,
                "quality_threshold": (
                    request.quality_threshold
                    if request
                    else float(DEFAULT_QUALITY_TOLERANCE)
                ),
                "identity_threshold": (
                    request.identity_threshold
                    if request
                    else float(DEFAULT_IDENTITY_TOLERANCE)
                ),
                "prompt": request.prompt if request else VISION_TEXT_PROMPT,
                "segmentation_found": False,
                "centroid": None,
                "inference_ms": None,
                "error": f"{type(exc).__name__}: {exc}",
                "source_image": request.source_image if request else None,
                "mask_image": None,
                "annotated_image": None,
                "completed_at": _utc_now(),
            }
            _atomic_write_json(job_path / RESULT_FILE, error_payload)
            log.exception("Inspection job %s failed", inspection_id)
            return True
        finally:
            try:
                (job_path / LOCK_FILE).unlink()
            except FileNotFoundError:
                pass

    def _acquire_lock(self, job_dir: Path) -> bool:
        lock_path = job_dir / LOCK_FILE
        if lock_path.exists():
            try:
                age_s = time.time() - lock_path.stat().st_mtime
            except OSError:
                return False
            if age_s <= self.stale_lock_s:
                return False
            log.warning("Removing stale inspection lock: %s", lock_path)
            try:
                lock_path.unlink()
            except OSError:
                return False

        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            os.write(
                descriptor,
                f"pid={os.getpid()} claimed_at={_utc_now()}\n".encode("utf-8"),
            )
        finally:
            os.close(descriptor)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch finished-print inspection jobs and write result.json files."
    )
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument(
        "--reference-prompt",
        default=VISION_TEXT_PROMPT,
        help="Prompt used while segmenting the reference catalogue.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process all currently pending jobs and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Delayed import keeps worker contract tests independent from CUDA/model loading.
    from inspection import ZeroShotInspector

    inspector = ZeroShotInspector()
    inspector.load_models()
    references = inspector.load_reference_directory(
        args.references,
        prompt=args.reference_prompt,
    )
    log.info("Registered %d reference images.", len(references))

    worker = InspectionWorker(
        jobs_dir=args.jobs_dir,
        inspector=inspector,
        poll_interval_s=args.poll_interval,
    )
    if args.once:
        processed = worker.run_pending()
        log.info("Processed %d pending inspection job(s).", processed)
        return 0

    try:
        worker.serve_forever()
    except KeyboardInterrupt:
        log.info("Inspection worker stopped by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
