"""Manual milestone: run the finished-print inspector on one saved image.

Run this before connecting anything to the printer, UCDavisGUI, or Tulip.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2

from config import (
    DEFAULT_IDENTITY_TOLERANCE,
    DEFAULT_QUALITY_TOLERANCE,
    VISION_TEXT_PROMPT,
)
from inspection import ZeroShotInspector

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one saved finished-print image using Grounded-SAM, DINOv2 and FAISS.",
    )
    parser.add_argument("--image", required=True, help="Finished-print image path.")
    parser.add_argument(
        "--references",
        required=True,
        help="Reference root. Layout: references/<part_id>/<image>.jpg",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for result.json, source.jpg, mask.png and annotated.jpg.",
    )
    parser.add_argument(
        "--expected-part",
        default=None,
        help="Part ID expected from the print queue. Omit for open-set identification.",
    )
    parser.add_argument("--prompt", default=VISION_TEXT_PROMPT)
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=float(DEFAULT_QUALITY_TOLERANCE),
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=float(DEFAULT_IDENTITY_TOLERANCE),
    )
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
    temp.replace(path)


def main() -> int:
    args = parse_args()
    source_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if frame is None:
        log.error("Could not read source image: %s", source_path)
        return 2

    inspector = ZeroShotInspector()
    inspector.load_models()
    references = inspector.load_reference_directory(
        args.references,
        prompt=args.prompt,
    )
    log.info("Registered %d reference images.", len(references))

    result, mask, annotated = inspector.inspect(
        frame,
        expected_part=args.expected_part,
        quality_threshold=args.quality_threshold,
        identity_threshold=args.identity_threshold,
        prompt=args.prompt,
    )

    cv2.imwrite(str(output_dir / "source.jpg"), frame)
    if mask is not None:
        cv2.imwrite(str(output_dir / "mask.png"), mask)
    if annotated is not None:
        cv2.imwrite(str(output_dir / "annotated.jpg"), annotated)
    write_json_atomic(output_dir / "result.json", result.to_dict())

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.status == "COMPLETE" else 3


if __name__ == "__main__":
    sys.exit(main())
