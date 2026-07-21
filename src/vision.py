"""SAM, DINOv2, FAISS and dual-camera processing for FrED."""

from __future__ import annotations

import logging
import threading
import time

# Import config before torch so the OpenMP environment variable is already set.
from config import (
    BOX_THRESHOLD,
    DINO_DIM,
    DINO_MODEL_NAME,
    GROUNDING_DINO_CHECKPOINT,
    GROUNDING_DINO_CONFIG,
    ROI_SIZE,
    SAM_CHECKPOINT,
    SHOW_SIDE_DEBUG,
    TEXT_THRESHOLD,
    VISION_TEXT_PROMPT,
    Z_GRASP_MIN,
)

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from groundingdino.util.inference import Model as DINOModel
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoImageProcessor, AutoModel

log = logging.getLogger(__name__)


class VisionSystem:
    """Owns the AI models, FAISS catalogue and background vision thread."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dino_processor = None
        self.dino_model = None
        self.grounding_dino = None
        self.sam_predictor = None
        self._models_loaded = False
        self._model_lock = threading.Lock()

        base_index = faiss.IndexFlatL2(DINO_DIM)
        self._faiss_index = faiss.IndexIDMap2(base_index)
        self._part_database: dict[int, str] = {}
        self._sample_count = 0
        self._feature_lock = threading.Lock()

        self._state_lock = threading.Lock()
        self._frame_top: np.ndarray | None = None
        self._frame_side: np.ndarray | None = None
        self._outputs: dict = {
            "x": 0,
            "y": 0,
            "z": Z_GRASP_MIN,
            "pieza": "Unknown",
            "distancia": 0.0,
            "mask_top": None,
            "roi_actual": None,
        }

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def sample_count(self) -> int:
        with self._feature_lock:
            return self._sample_count

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    def load_models(self) -> None:
        """Load DINOv2, Grounding DINO and SAM once."""
        if self._models_loaded:
            return

        log.info("Loading DINOv2…")
        self.dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        self.dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME)
        self.dino_model.eval()

        log.info("Loading Grounded-SAM…")
        log.info("Using device: %s", self.device)
        self.grounding_dino = DINOModel(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.device,
        )

        sam_model = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
        sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(sam_model)

        self._models_loaded = True
        log.info("All AI models loaded.")

    def start(self) -> None:
        """Load models and start the background processing thread."""
        self.load_models()
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._processing_loop,
            name="fred-vision",
            daemon=True,
        )
        self._thread.start()
        log.info("AI thread started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if SHOW_SIDE_DEBUG:
            try:
                cv2.destroyWindow("DEBUG CAMARA LATERAL")
            except cv2.error:
                pass

    def set_frames(
        self, top_frame: np.ndarray, side_frame: np.ndarray | None
    ) -> None:
        """Publish the newest camera frames for the AI thread."""
        with self._state_lock:
            self._frame_top = top_frame.copy()
            self._frame_side = side_frame.copy() if side_frame is not None else None

    def snapshot(self) -> dict:
        """Return a thread-safe snapshot of the latest AI result."""
        with self._state_lock:
            return dict(self._outputs)

    def register_sample(self, part_name: str) -> int:
        """Add the latest clean ROI to the FAISS catalogue."""
        if not part_name or part_name == "Unknown":
            raise ValueError("A valid part name is required.")

        snapshot = self.snapshot()
        roi = snapshot["roi_actual"]
        if roi is None:
            raise RuntimeError("Waiting for a valid SAM detection.")

        vector = self.extract_embedding(roi)
        with self._feature_lock:
            faiss_id = np.array([self._sample_count], dtype=np.int64)
            self._faiss_index.add_with_ids(vector, faiss_id)
            self._part_database[self._sample_count] = part_name
            self._sample_count += 1
            return self._sample_count

    def extract_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        """Extract the DINOv2 CLS embedding from a BGR image."""
        if not self._models_loaded:
            raise RuntimeError("Vision models are not loaded.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        inputs = self.dino_processor(images=image_pil, return_tensors="pt")

        with self._model_lock:
            with torch.no_grad():
                outputs = self.dino_model(**inputs)

        return (
            outputs.last_hidden_state[:, 0, :]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def segment_with_sam(
        self,
        image_rgb: np.ndarray,
        text_prompt: str = VISION_TEXT_PROMPT,
    ) -> np.ndarray | None:
        """Detect the prompted part with Grounding DINO and segment it with SAM."""
        if not self._models_loaded:
            raise RuntimeError("Vision models are not loaded.")

        with self._model_lock:
            detections, _ = self.grounding_dino.predict_with_caption(
                image=image_rgb,
                caption=text_prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
            if len(detections.xyxy) == 0:
                return None

            best_idx = int(np.argmax(detections.confidence))
            sam_box = detections.xyxy[best_idx]
            self.sam_predictor.set_image(image_rgb)
            masks, _, _ = self.sam_predictor.predict(
                box=sam_box,
                multimask_output=False,
            )

        return (masks[0] * 255).astype(np.uint8)

    def _update_outputs(self, **kwargs) -> None:
        with self._state_lock:
            self._outputs.update(kwargs)

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._state_lock:
                    top_frame = self._frame_top
                    side_frame = self._frame_side

                if top_frame is None:
                    time.sleep(0.05)
                    continue

                top_frame = top_frame.copy()
                side_frame = side_frame.copy() if side_frame is not None else None
                height, width = top_frame.shape[:2]

                rgb_top = cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB)
                mask_top = self.segment_with_sam(rgb_top)

                with self._state_lock:
                    center_x = self._outputs["x"]
                    center_y = self._outputs["y"]
                    z_mm = self._outputs["z"]

                clean_roi = None
                if mask_top is not None:
                    moments = cv2.moments(mask_top)
                    if moments["m00"] != 0:
                        center_x = int(moments["m10"] / moments["m00"])
                        center_y = int(moments["m01"] / moments["m00"])

                    clean_image = cv2.bitwise_and(
                        top_frame,
                        top_frame,
                        mask=mask_top,
                    )
                    half = ROI_SIZE // 2
                    clipped_x = int(np.clip(center_x, half, width - half))
                    clipped_y = int(np.clip(center_y, half, height - half))
                    x1, y1 = clipped_x - half, clipped_y - half
                    clean_roi = clean_image[
                        y1 : y1 + ROI_SIZE,
                        x1 : x1 + ROI_SIZE,
                    ].copy()

                if side_frame is not None:
                    hsv_side = cv2.cvtColor(side_frame, cv2.COLOR_BGR2HSV)
                    mask_red1 = cv2.inRange(
                        hsv_side,
                        np.array([0, 100, 50]),
                        np.array([10, 255, 255]),
                    )
                    mask_red2 = cv2.inRange(
                        hsv_side,
                        np.array([170, 100, 50]),
                        np.array([180, 255, 255]),
                    )
                    side_mask = cv2.bitwise_or(mask_red1, mask_red2)
                    kernel = np.ones((5, 5), np.uint8)
                    side_mask = cv2.morphologyEx(
                        side_mask,
                        cv2.MORPH_OPEN,
                        kernel,
                    )

                    if SHOW_SIDE_DEBUG:
                        cv2.imshow("DEBUG CAMARA LATERAL", side_mask)
                        cv2.waitKey(1)

                    side_moments = cv2.moments(side_mask)
                    if side_moments["m00"] != 0:
                        side_y = int(side_moments["m01"] / side_moments["m00"])
                        log.debug("Side camera Y pixel: %d", side_y)
                        z_mm = float(
                            np.clip(-0.5007 * side_y + 297.76, 40.0, 220.0)
                        )

                distance = 0.0
                part_name = "Unknown"
                if clean_roi is not None:
                    with self._feature_lock:
                        has_samples = self._sample_count > 0

                    if has_samples:
                        current_vector = self.extract_embedding(clean_roi)
                        with self._feature_lock:
                            distances, indices = self._faiss_index.search(
                                current_vector,
                                1,
                            )
                            distance = float(distances[0][0])
                            faiss_id = int(indices[0][0])
                            part_name = self._part_database.get(
                                faiss_id,
                                "Unknown",
                            )

                self._update_outputs(
                    x=center_x,
                    y=center_y,
                    z=z_mm,
                    mask_top=mask_top,
                    roi_actual=clean_roi,
                    distancia=distance,
                    pieza=part_name,
                )
            except Exception as exc:
                log.error(
                    "AI thread error (will continue): %s",
                    exc,
                    exc_info=True,
                )

            time.sleep(0.01)
