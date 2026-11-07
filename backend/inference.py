from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOBrainTumorEngine:
    def __init__(self, model_path: Path, conf: float = 0.25, detection_model_path: Path | None = None):
        self.model_path = Path(model_path)
        self.detection_model_path = Path(detection_model_path) if detection_model_path else self.model_path.parent / "detection.pt"
        self.conf = conf
        self.model = None
        self.detector = None
        self.device = "CPU"
        self.loaded = False
        self.detector_loaded = False
        self._try_load()

    def _try_load(self) -> None:
        if not str(self.model_path) or str(self.model_path) == ".":
            return
        if not self.model_path.exists():
            return
        try:
            self.model = YOLO(str(self.model_path))
            self.loaded = True
            try:
                self.device = str(next(self.model.model.parameters()).device)
            except Exception:
                self.device = "CPU"
        except Exception:
            self.model = None
            self.loaded = False
        if self.detection_model_path.exists():
            try:
                self.detector = YOLO(str(self.detection_model_path))
                self.detector_loaded = True
            except Exception:
                self.detector = None
                self.detector_loaded = False

    def info(self) -> dict[str, Any]:
        names = {}
        if self.model is not None:
            raw_names = getattr(self.model, "names", {})
            if isinstance(raw_names, list):
                names = {i: n for i, n in enumerate(raw_names)}
            else:
                names = raw_names
        return {
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "detection_model_loaded": self.detector_loaded,
            "detection_model_path": str(self.detection_model_path),
            "device": self.device,
            "confidence_threshold": self.conf,
            "classes": names,
        }

    def predict(self, image_path: Path) -> dict[str, Any]:
        if not self.loaded or self.model is None:
            raise RuntimeError(
                f"YOLO model not loaded. Put trained weights at {self.model_path}."
            )

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError("Could not read the supplied image")

        processed_views = self._build_classification_views(frame)
        results = self.model.predict(source=processed_views, conf=self.conf, verbose=False)
        if not results:
            raise RuntimeError("Model did not return any prediction results")

        result = results[0]
        names = getattr(self.model, "names", {})
        probs = getattr(result, "probs", None)

        if probs is not None:
            averaged_probs = self._average_probabilities(results)
            if averaged_probs is not None:
                top1 = int(np.argmax(averaged_probs))
                top1conf = float(averaged_probs[top1])
                probability_vector = averaged_probs
            else:
                top1 = int(probs.top1)
                top1conf = float(probs.top1conf)
                probability_vector = probs.data.detach().cpu().numpy().astype(np.float32)
            class_probabilities = []
            for class_index, class_probability in enumerate(probability_vector):
                class_name = names[class_index] if isinstance(names, dict) else names[class_index]
                class_probabilities.append({
                    "class_id": int(class_index),
                    "class": str(class_name),
                    "probability": round(float(class_probability), 6),
                })
            class_probabilities.sort(key=lambda item: item["probability"], reverse=True)
            label = names[top1] if isinstance(names, dict) else names[top1]
            normalized_label = str(label).strip().lower().replace(" ", "")
            box = None
            detections = []
            localization_method = "unavailable"
            if normalized_label not in {"notumor", "no_tumor"}:
                detector_detections = self._predict_detector(frame)
                if detector_detections:
                    detections = detector_detections
                    localization_method = "detector"
                    best_detection = max(detector_detections, key=lambda d: d["confidence"])
                    label = best_detection["class"]
                    normalized_label = str(label).strip().lower().replace(" ", "")
                    top1conf = float(best_detection["confidence"])
                    box_data = best_detection["bbox"]
                    box = (
                        int(box_data["x1"]),
                        int(box_data["y1"]),
                        int(box_data["x2"]),
                        int(box_data["y2"]),
                    )
                else:
                    box = self._infer_bbox(frame, label=str(label))
                    if box is None:
                        box = self._infer_occlusion_bbox(frame, class_id=top1, baseline_conf=top1conf)
                    if box is not None:
                        localization_method = "heuristic_estimate"
                if box is not None:
                    x1, y1, x2, y2 = box
                    if not detections:
                        detections.append(
                            {
                                "class_id": top1,
                                "class": str(label),
                                "confidence": round(top1conf, 4),
                                "source": localization_method,
                                "bbox": {
                                    "x1": round(float(x1), 2),
                                    "y1": round(float(y1), 2),
                                    "x2": round(float(x2), 2),
                                    "y2": round(float(y2), 2),
                                },
                            }
                        )
            return {
                "status": "success",
                "task": "classification",
                "tumor_detected": normalized_label not in {"notumor", "no_tumor"},
                "predicted_type": str(label),
                "predicted_class_id": int(top1),
                "best_confidence": round(top1conf, 4),
                "class_probabilities": class_probabilities,
                "top_predictions": class_probabilities[:4],
                "detections": detections,
                "localization_available": bool(detections),
                "localization_method": localization_method,
                "annotated_image": self._save_annotated_image(
                    frame,
                    image_path,
                    label=str(label),
                    confidence=top1conf,
                    box=box,
                ),
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
            }

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                label = names[cls_id] if isinstance(names, dict) else names[cls_id]
                detections.append(
                    {
                        "class_id": cls_id,
                        "class": str(label),
                        "confidence": round(confidence, 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        },
                    }
                )

        annotated_name = f"{image_path.stem}_annotated.jpg"
        annotated_path = image_path.parent.parent / "outputs" / annotated_name
        annotated_frame = self._build_annotated_frame(frame, detections, result)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated_frame)

        best = max(detections, key=lambda d: d["confidence"], default=None)
        return {
            "status": "success",
            "task": "detection",
            "tumor_detected": bool(detections),
            "predicted_type": best["class"] if best else None,
            "best_confidence": best["confidence"] if best else None,
            "detections": detections,
            "localization_available": bool(detections),
            "localization_method": "detector" if detections else "unavailable",
            "annotated_image": annotated_name,
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0]),
        }

    def _build_classification_views(self, frame: np.ndarray) -> list[np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        views = [rgb]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        views.append(equalized_rgb)

        h, w = gray.shape[:2]
        crop_margin_y = max(1, int(h * 0.06))
        crop_margin_x = max(1, int(w * 0.06))
        cropped = frame[crop_margin_y:h - crop_margin_y, crop_margin_x:w - crop_margin_x]
        if cropped.size > 0:
            views.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        views.append(cv2.flip(rgb, 1))
        return views

    def _average_probabilities(self, results: list[Any]) -> np.ndarray | None:
        vectors: list[np.ndarray] = []
        for result in results:
            probs = getattr(result, "probs", None)
            if probs is None:
                continue
            vector = probs.data.detach().cpu().numpy().astype(np.float32)
            if vector.size:
                vectors.append(vector)
        if not vectors:
            return None
        stacked = np.stack(vectors, axis=0)
        return stacked.mean(axis=0)

    def _predict_detector(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.detector is None or not self.detector_loaded:
            return []

        try:
            results = self.detector.predict(source=frame, conf=max(0.12, self.conf * 0.5), verbose=False)
        except Exception:
            return []
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(self.detector, "names", {})
        detections: list[dict[str, Any]] = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = names[cls_id] if isinstance(names, dict) else names[cls_id]
            normalized_label = str(label).strip().lower().replace(" ", "").replace("_", "")
            if normalized_label in {"notumor", "notumour", "no_tumor", "notumor"}:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": cls_id,
                    "class": self._normalize_display_label(str(label)),
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def _normalize_display_label(self, label: str) -> str:
        normalized = label.strip().lower().replace("_", " ")
        if normalized in {"no tumor", "notumor"}:
            return "notumor"
        return normalized

    def _save_classification_image(self, frame: np.ndarray, image_path: Path, label: str, confidence: float | None = None) -> str:
        annotated_name = f"{image_path.stem}_annotated.jpg"
        annotated_path = image_path.parent.parent / "outputs" / annotated_name
        annotated_frame = frame.copy()
        text = label if confidence is None else f"{label} {confidence:.2f}"
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            f"Classification: {text}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated_frame)
        return annotated_name

    def _infer_occlusion_bbox(self, frame: np.ndarray, class_id: int, baseline_conf: float) -> tuple[int, int, int, int] | None:
        if self.model is None or baseline_conf <= 0:
            return None

        h, w = frame.shape[:2]
        patch = max(40, min(h, w) // 7)
        stride = max(16, patch // 3)
        samples: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        fill_color = tuple(int(v) for v in np.median(frame.reshape(-1, 3), axis=0))

        for y in range(0, max(1, h - patch + 1), stride):
            for x in range(0, max(1, w - patch + 1), stride):
                x2 = min(w, x + patch)
                y2 = min(h, y + patch)
                masked = frame.copy()
                cv2.rectangle(masked, (x, y), (x2, y2), fill_color, -1)
                samples.append(masked)
                boxes.append((x, y, x2, y2))

        if not samples:
            return None

        try:
            results = self.model.predict(source=samples, verbose=False, batch=16)
        except Exception:
            return None

        heat = np.zeros((h, w), dtype=np.float32)
        for result, box in zip(results, boxes):
            probs = getattr(result, "probs", None)
            if probs is None:
                continue
            confidences = probs.data.detach().cpu().numpy()
            if class_id >= len(confidences):
                continue
            drop = baseline_conf - float(confidences[class_id])
            if drop > max(0.015, baseline_conf * 0.025):
                x1, y1, x2, y2 = box
                heat[y1:y2, x1:x2] += float(drop)

        if not np.any(heat):
            return None

        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(3, patch / 7.0))
        threshold = max(float(np.percentile(heat[heat > 0], 82)), float(np.max(heat)) * 0.45)
        mask = (heat >= threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 12 or bh < 12:
                continue
            component_mask = np.zeros_like(mask)
            cv2.drawContours(component_mask, [contour], -1, 255, -1)
            score = float(np.sum(heat[component_mask > 0]))
            candidates.append((score, (x, y, x + bw, y + bh)))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        x1, y1, x2, y2 = candidates[0][1]
        pad = max(6, patch // 8)
        box = (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad),
            min(h, y2 + pad),
        )
        return self._tighten_bbox_to_bright_region(frame, box) or box

    def _tighten_bbox_to_bright_region(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        _, mask = cv2.threshold(blur, int(np.percentile(blur, 88)), 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        roi_area = float(roi.shape[0] * roi.shape[1])
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < roi_area * 0.01 or area > roi_area * 0.7:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 10 or bh < 10:
                continue
            candidates.append((area, (x1 + x, y1 + y, x1 + x + bw, y1 + y + bh)))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _save_annotated_image(
        self,
        frame: np.ndarray,
        image_path: Path,
        label: str = "Tumor",
        confidence: float | None = None,
        box: tuple[int, int, int, int] | None = None,
    ) -> str:
        annotated_name = f"{image_path.stem}_annotated.jpg"
        annotated_path = image_path.parent.parent / "outputs" / annotated_name
        annotated_frame = self._build_annotated_frame(
            frame,
            [],
            None,
            label=label,
            confidence=confidence,
            forced_box=box,
        )
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated_frame)
        return annotated_name

    def _build_annotated_frame(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        result: Any,
        label: str = "Tumor",
        confidence: float | None = None,
        forced_box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        if detections:
            annotated = result.plot(line_width=2, labels=True, conf=True)
            if annotated is not None:
                return annotated

        box = forced_box if forced_box is not None else self._infer_bbox(frame, label=label)
        annotated = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = label if confidence is None else f"{label} {confidence:.2f}"
            cv2.putText(
                annotated,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _infer_bbox(self, frame: np.ndarray, label: str | None = None) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape[:2]
        image_area = float(h * w)

        # Find the brain region first, then search for a smaller bright lesion inside it.
        _, brain_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        brain_contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not brain_contours:
            return None

        brain_contour = max(brain_contours, key=cv2.contourArea)
        brain_area = cv2.contourArea(brain_contour)
        if brain_area <= 0:
            return None

        bx, by, bw, bh = cv2.boundingRect(brain_contour)
        brain_roi = gray[by:by + bh, bx:bx + bw]
        if brain_roi.size == 0:
            return None

        normalized_label = str(label or "").strip().lower().replace(" ", "")
        if normalized_label == "pituitary":
            return self._infer_pituitary_bbox(gray, bx, by, bw, bh)
        if normalized_label == "meningioma":
            return self._infer_meningioma_bbox(gray, bx, by, bw, bh)
        if normalized_label == "glioma":
            return self._infer_glioma_bbox(gray, bx, by, bw, bh)

        # Generic fallback for the remaining classes.
        generic = self._infer_generic_bbox(brain_roi, bx, by)
        return generic or self._fallback_tumor_bbox(gray, normalized_label, bx, by, bw, bh)

    def _infer_generic_bbox(self, brain_roi: np.ndarray, bx: int, by: int) -> tuple[int, int, int, int] | None:
        roi_blur = cv2.GaussianBlur(brain_roi, (5, 5), 0)
        roi_thresh_level = int(np.percentile(roi_blur, 94))
        _, bright = cv2.threshold(roi_blur, roi_thresh_level, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        roi_area = float(brain_roi.shape[0] * brain_roi.shape[1])
        roi_cx = brain_roi.shape[1] / 2.0
        roi_cy = brain_roi.shape[0] / 2.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < roi_area * 0.003 or area > roi_area * 0.18:
                continue
            x, y, bw2, bh2 = cv2.boundingRect(contour)
            if bw2 < 10 or bh2 < 10:
                continue
            fill_ratio = area / float(max(1, bw2 * bh2))
            if fill_ratio < 0.12:
                continue
            cx = x + bw2 / 2.0
            cy = y + bh2 / 2.0
            center_dist = np.hypot((cx - roi_cx) / max(1.0, roi_cx), (cy - roi_cy) / max(1.0, roi_cy))
            center_bias = max(0.0, 1.4 - center_dist)
            compactness = area / float(max(1, (bw2 + bh2) ** 2))
            score = area * (1.0 + center_bias) * (1.0 + compactness)
            candidates.append((score, (bx + x, by + y, bx + x + bw2, by + y + bh2)))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]

        return None

    def _infer_glioma_bbox(self, gray: np.ndarray, bx: int, by: int, bw: int, bh: int) -> tuple[int, int, int, int] | None:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, brain_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._fallback_tumor_bbox(gray, "glioma", bx, by, bw, bh)

        brain_contour = max(contours, key=cv2.contourArea)
        filled_brain = np.zeros_like(gray)
        cv2.drawContours(filled_brain, [brain_contour], -1, 255, -1)
        inner_brain = cv2.erode(filled_brain, np.ones((11, 11), np.uint8), iterations=1)
        brain_values = blur[inner_brain > 0]
        if brain_values.size == 0:
            return self._fallback_tumor_bbox(gray, "glioma", bx, by, bw, bh)

        distance = cv2.distanceTransform(inner_brain, cv2.DIST_L2, 5)
        image_area = float(gray.shape[0] * gray.shape[1])
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []

        for percentile in (99, 98, 97, 96, 95, 94, 93, 92, 90):
            _, bright = cv2.threshold(blur, int(np.percentile(brain_values, percentile)), 255, cv2.THRESH_BINARY)
            bright = cv2.bitwise_and(bright, bright, mask=inner_brain)
            bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

            bright_contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in bright_contours:
                area = cv2.contourArea(contour)
                if area < image_area * 0.00045 or area > image_area * 0.09:
                    continue
                x, y, bw2, bh2 = cv2.boundingRect(contour)
                if bw2 < 12 or bh2 < 12:
                    continue
                if bw2 > gray.shape[1] * 0.48 or bh2 > gray.shape[0] * 0.48:
                    continue
                cx = x + bw2 / 2.0
                cy = y + bh2 / 2.0
                if cx < bx + bw * 0.16 or cy < by + bh * 0.10 or cy > by + bh * 0.82:
                    continue
                fill_ratio = area / float(max(1, bw2 * bh2))
                if fill_ratio < 0.10:
                    continue

                pad = 10
                px1 = max(0, x - pad)
                py1 = max(0, y - pad)
                px2 = min(gray.shape[1], x + bw2 + pad)
                py2 = min(gray.shape[0], y + bh2 + pad)
                padded = gray[py1:py2, px1:px2]
                if padded.size == 0:
                    continue
                border = np.concatenate([padded[0, :], padded[-1, :], padded[:, 0], padded[:, -1]])
                lesion = gray[y:y + bh2, x:x + bw2]
                contrast = float(np.mean(lesion)) - float(np.mean(border))
                if contrast < 5:
                    continue

                depth = float(np.mean(distance[y:y + bh2, x:x + bw2]))
                if depth < 6:
                    continue
                compactness = min(bw2, bh2) / float(max(1, max(bw2, bh2)))
                lesion_size = np.sqrt(area)
                score = lesion_size * (1.0 + max(0.0, contrast) / 18.0) * (1.0 + fill_ratio) * (1.0 + compactness * 1.6)
                score *= 1.0 + min(depth / 45.0, 1.0)
                score *= 1.0 + (99 - percentile) * 0.015
                candidates.append((score, (max(0, x - 6), max(0, y - 6), min(gray.shape[1], x + bw2 + 6), min(gray.shape[0], y + bh2 + 6))))

        compact_box = self._infer_compact_enhancing_bbox(gray, inner_brain, bx, by, bw, bh)
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_box = candidates[0][1]
            if compact_box is not None and self._should_prefer_compact_box(best_box, compact_box, bx, by, bw, bh):
                return compact_box
            return best_box

        return self._find_brightest_patch_bbox(gray, (bx, by, bx + bw, by + bh), "glioma") or compact_box or self._fallback_tumor_bbox(
            gray,
            "glioma",
            bx,
            by,
            bw,
            bh,
        )

    def _infer_compact_enhancing_bbox(
        self,
        gray: np.ndarray,
        inner_brain: np.ndarray,
        bx: int,
        by: int,
        bw: int,
        bh: int,
    ) -> tuple[int, int, int, int] | None:
        small_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        large_blur = cv2.GaussianBlur(gray, (31, 31), 0)
        contrast_map = cv2.subtract(small_blur, large_blur)
        contrast_map = cv2.bitwise_and(contrast_map, contrast_map, mask=inner_brain)
        values = contrast_map[inner_brain > 0]
        if values.size == 0:
            return None

        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        distance = cv2.distanceTransform(inner_brain, cv2.DIST_L2, 5) if np.any(inner_brain) else None
        for percentile in (99.5, 99, 98.5, 98, 97.5, 97, 96):
            threshold = int(np.percentile(values, percentile))
            _, mask = cv2.threshold(contrast_map, threshold, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 65 or area > 950:
                    continue
                x, y, bw2, bh2 = cv2.boundingRect(contour)
                if bw2 < 16 or bh2 < 14 or bw2 > 60 or bh2 > 60:
                    continue
                cx = x + bw2 / 2.0
                cy = y + bh2 / 2.0
                if cx < bx + bw * 0.18 or cy < by + bh * 0.08 or cy > by + bh * 0.45:
                    continue
                if cx > bx + bw * 0.85 and cy < by + bh * 0.25:
                    continue
                compactness = min(bw2, bh2) / float(max(1, max(bw2, bh2)))
                if compactness < 0.45:
                    continue
                fill_ratio = area / float(max(1, bw2 * bh2))
                if fill_ratio < 0.08:
                    continue

                patch = contrast_map[y:y + bh2, x:x + bw2]
                local_contrast = float(np.mean(patch)) if patch.size else 0.0
                if local_contrast < 8:
                    continue
                depth = float(np.mean(distance[y:y + bh2, x:x + bw2])) if distance is not None else 0.0
                if depth < 7:
                    continue
                lesion_patch = gray[y:y + bh2, x:x + bw2]
                if lesion_patch.size == 0:
                    continue
                center_margin_x = max(1, int(bw2 * 0.25))
                center_margin_y = max(1, int(bh2 * 0.25))
                center = lesion_patch[center_margin_y:bh2 - center_margin_y, center_margin_x:bw2 - center_margin_x]
                border_values = np.concatenate(
                    [
                        lesion_patch[0, :],
                        lesion_patch[-1, :],
                        lesion_patch[:, 0],
                        lesion_patch[:, -1],
                    ]
                )
                ring_score = max(0.0, float(np.mean(border_values)) - float(np.mean(center))) if center.size else 0.0
                upper_bias = 1.0 + max(0.0, (by + bh * 0.45 - cy) / max(1.0, bh)) * 1.8
                score = local_contrast * (1.0 + compactness) * (1.0 + fill_ratio) * (1.0 + ring_score / 24.0) * upper_bias
                score *= 1.0 + min(depth / 35.0, 1.0)
                score *= 1.0 + max(0.0, 99.5 - percentile) * 0.03
                pad = 8
                candidates.append(
                    (
                        score,
                        (
                            max(0, x - pad),
                            max(0, y - pad),
                            min(gray.shape[1], x + bw2 + pad),
                            min(gray.shape[0], y + bh2 + pad),
                        ),
                    )
                )

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _should_prefer_compact_box(
        self,
        best_box: tuple[int, int, int, int],
        compact_box: tuple[int, int, int, int],
        bx: int,
        by: int,
        bw: int,
        bh: int,
    ) -> bool:
        x1, y1, x2, y2 = best_box
        cx1, cy1, cx2, cy2 = compact_box
        best_cy = (y1 + y2) / 2.0
        compact_cy = (cy1 + cy2) / 2.0
        best_area = float(max(1, (x2 - x1) * (y2 - y1)))
        compact_area = float(max(1, (cx2 - cx1) * (cy2 - cy1)))
        brain_area = float(max(1, bw * bh))
        lower_best = best_cy > by + bh * 0.50
        upper_compact = compact_cy < by + bh * 0.45
        much_smaller = compact_area < best_area * 0.65
        best_is_not_large_lesion = best_area < brain_area * 0.05
        return lower_best and upper_compact and much_smaller and best_is_not_large_lesion

    def _infer_meningioma_bbox(self, gray: np.ndarray, bx: int, by: int, bw: int, bh: int) -> tuple[int, int, int, int] | None:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        upper_box = self._infer_upper_meningioma_bbox(gray, blur)
        if upper_box is not None:
            return upper_box

        _, brain_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        brain_contour = max(contours, key=cv2.contourArea)
        filled_brain = np.zeros_like(gray)
        cv2.drawContours(filled_brain, [brain_contour], -1, 255, -1)
        inner_brain = cv2.erode(filled_brain, np.ones((13, 13), np.uint8), iterations=1)
        brain_values = blur[inner_brain > 0]
        if brain_values.size == 0:
            return None

        distance = cv2.distanceTransform(inner_brain, cv2.DIST_L2, 5)
        image_area = float(gray.shape[0] * gray.shape[1])
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []

        for percentile in (99, 98, 97, 96, 95, 94, 93, 92, 90, 88):
            _, bright = cv2.threshold(blur, int(np.percentile(brain_values, percentile)), 255, cv2.THRESH_BINARY)
            bright = cv2.bitwise_and(bright, bright, mask=inner_brain)
            bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            bright_contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in bright_contours:
                area = cv2.contourArea(contour)
                if area < image_area * 0.00035 or area > image_area * 0.028:
                    continue
                x, y, bw2, bh2 = cv2.boundingRect(contour)
                if bw2 < 10 or bh2 < 10:
                    continue
                if bw2 > gray.shape[1] * 0.45 or bh2 > gray.shape[0] * 0.45:
                    continue
                aspect = bw2 / float(max(1, bh2))
                if aspect < 0.52 or aspect > 2.8:
                    continue
                fill_ratio = area / float(max(1, bw2 * bh2))
                if fill_ratio < 0.12:
                    continue
                if (y + bh2 / 2.0) > gray.shape[0] * 0.78:
                    continue

                pad = 8
                px1 = max(0, x - pad)
                py1 = max(0, y - pad)
                px2 = min(gray.shape[1], x + bw2 + pad)
                py2 = min(gray.shape[0], y + bh2 + pad)
                padded = gray[py1:py2, px1:px2]
                if padded.size == 0:
                    continue
                border = np.concatenate([padded[0, :], padded[-1, :], padded[:, 0], padded[:, -1]])
                lesion_mean = float(np.mean(gray[y:y + bh2, x:x + bw2]))
                contrast = lesion_mean - float(np.mean(border))
                if contrast < 12:
                    continue

                brain_depth = float(np.mean(distance[y:y + bh2, x:x + bw2]))
                if brain_depth < 12:
                    continue
                compactness = min(bw2, bh2) / float(max(1, max(bw2, bh2)))
                score = area * (1.0 + max(0.0, contrast) / 35.0) * (1.0 + fill_ratio * 2.0) * (1.0 + compactness)
                score *= 1.0 + min(brain_depth / 60.0, 1.0)
                score *= 1.0 + (99 - percentile) * 0.02
                candidates.append((score, (x, y, x + bw2, y + bh2)))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            compact_box = self._infer_compact_enhancing_bbox(gray, inner_brain, bx, by, bw, bh)
            best_box = candidates[0][1]
            if compact_box is not None and self._should_prefer_compact_box(best_box, compact_box, bx, by, bw, bh):
                return compact_box
            return best_box

        generic = self._infer_generic_bbox(gray[by:by + bh, bx:bx + bw], bx, by)
        compact_box = self._infer_compact_enhancing_bbox(gray, inner_brain, bx, by, bw, bh)
        return generic or self._find_brightest_patch_bbox(gray, (bx, by, bx + bw, by + bh), "meningioma") or compact_box or self._fallback_tumor_bbox(
            gray,
            "meningioma",
            bx,
            by,
            bw,
            bh,
        )

    def _infer_upper_meningioma_bbox(self, gray: np.ndarray, blur: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = gray.shape[:2]
        search_mask = np.zeros_like(gray)
        search_mask[int(h * 0.08):int(h * 0.55), int(w * 0.18):int(w * 0.82)] = 255
        values = blur[search_mask > 0]
        if values.size == 0:
            return None

        image_area = float(h * w)
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for percentile in (92, 90, 88, 85):
            _, mask = cv2.threshold(blur, int(np.percentile(values, percentile)), 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, mask, mask=search_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image_area * 0.0015 or area > image_area * 0.04:
                    continue
                x, y, bw2, bh2 = cv2.boundingRect(contour)
                if bw2 < 18 or bh2 < 18:
                    continue
                aspect = bw2 / float(max(1, bh2))
                if aspect < 0.55 or aspect > 2.4:
                    continue
                fill_ratio = area / float(max(1, bw2 * bh2))
                if fill_ratio < 0.45:
                    continue
                score = area * fill_ratio * (1.0 + max(0, percentile - 85) * 0.04)
                pad = 6
                candidates.append(
                    (
                        score,
                        (
                            max(0, x - pad),
                            max(0, y - pad),
                            min(w, x + bw2 + pad),
                            min(h, y + bh2 + pad),
                        ),
                    )
                )

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _infer_pituitary_bbox(self, gray: np.ndarray, bx: int, by: int, bw: int, bh: int) -> tuple[int, int, int, int] | None:
        h, w = gray.shape[:2]

        # Pituitary tumors usually appear near the sella/pituitary region:
        # central-lower in coronal and central-middle in sagittal images.
        # Use a fixed center-weighted search region so skull-edge contours do not
        # pull the box toward the border.
        x1 = int(max(0, w * 0.22))
        x2 = int(min(w, w * 0.78))
        y1 = int(max(0, h * 0.18))
        y2 = int(min(h, h * 0.82))

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        thresh_level = int(np.percentile(roi_blur, 90))
        _, mask = cv2.threshold(roi_blur, thresh_level, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        roi_area = float(roi.shape[0] * roi.shape[1])
        roi_cx = roi.shape[1] / 2.0
        roi_cy = roi.shape[0] / 2.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < roi_area * 0.004 or area > roi_area * 0.35:
                continue
            x, y, bw2, bh2 = cv2.boundingRect(contour)
            if bw2 < 8 or bh2 < 8:
                continue
            fill_ratio = area / float(max(1, bw2 * bh2))
            if fill_ratio < 0.10:
                continue
            cx = x + bw2 / 2.0
            cy = y + bh2 / 2.0
            center_dist = np.hypot((cx - roi_cx) / max(1.0, roi_cx), (cy - roi_cy) / max(1.0, roi_cy))
            score = area * (1.35 - min(center_dist, 1.0)) * (1.0 + fill_ratio)
            candidates.append((score, (x1 + x, y1 + y, x1 + x + bw2, y1 + y + bh2)))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            compact_box = self._infer_compact_enhancing_bbox(gray, self._make_inner_brain_mask(gray), x1, y1, x2 - x1, y2 - y1)
            best_box = candidates[0][1]
            if compact_box is not None and self._should_prefer_compact_box(best_box, compact_box, x1, y1, x2 - x1, y2 - y1):
                return compact_box
            return best_box

        inner_brain = self._make_inner_brain_mask(gray)
        return self._find_brightest_patch_bbox(gray, (x1, y1, x2, y2), "pituitary") or self._infer_compact_enhancing_bbox(gray, inner_brain, x1, y1, x2 - x1, y2 - y1) or self._fallback_tumor_bbox(gray, "pituitary", bx=0, by=0, bw=w, bh=h)

    def _make_inner_brain_mask(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, brain_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(gray)
        filled_brain = np.zeros_like(gray)
        cv2.drawContours(filled_brain, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        return cv2.erode(filled_brain, np.ones((11, 11), np.uint8), iterations=1)

    def _find_brightest_patch_bbox(self, gray: np.ndarray, search_box: tuple[int, int, int, int], label: str) -> tuple[int, int, int, int] | None:
        if gray.size == 0:
            return None

        x1, y1, x2, y2 = search_box
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray_roi = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray_roi.shape[:2]
        if h < 20 or w < 20:
            return None

        blur = cv2.GaussianBlur(gray_roi, (7, 7), 0)
        _, brain_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        brain_mask = cv2.erode(brain_mask, np.ones((7, 7), np.uint8), iterations=1)

        patch_h = max(20, int(h * 0.16))
        patch_w = max(20, int(w * 0.16))
        step_y = max(8, patch_h // 4)
        step_x = max(8, patch_w // 4)
        best_score = None
        best_box = None
        distance = cv2.distanceTransform(brain_mask, cv2.DIST_L2, 5) if np.any(brain_mask) else None
        target_map = {
            "pituitary": (0.50, 0.48),
            "meningioma": (0.60, 0.34),
            "glioma": (0.56, 0.42),
        }
        tx, ty = target_map.get(str(label).strip().lower().replace(" ", ""), (0.50, 0.50))
        target_cx = w * tx
        target_cy = h * ty

        for y in range(0, max(1, h - patch_h + 1), step_y):
            for x in range(0, max(1, w - patch_w + 1), step_x):
                mask_patch = brain_mask[y:y + patch_h, x:x + patch_w]
                if mask_patch.size == 0:
                    continue
                coverage = float(np.mean(mask_patch > 0))
                if coverage < 0.5:
                    continue
                patch = blur[y:y + patch_h, x:x + patch_w]
                mean = float(np.mean(patch))
                std = float(np.std(patch))
                maxv = float(np.max(patch))
                depth = float(np.mean(distance[y:y + patch_h, x:x + patch_w])) if distance is not None else 0.0
                cx = x + patch_w / 2.0
                cy = y + patch_h / 2.0
                center_dist = np.hypot((cx - target_cx) / max(1.0, w), (cy - target_cy) / max(1.0, h))
                score = mean * 1.15 + std * 0.55 + maxv * 0.18
                score *= 1.0 + min(depth / 40.0, 1.2)
                score *= 1.0 + (coverage - 0.5) * 0.8
                score *= 1.25 - min(center_dist, 1.0)
                if best_score is None or score > best_score:
                    best_score = score
                    best_box = (x1 + x, y1 + y, x1 + min(w, x + patch_w), y1 + min(h, y + patch_h))

        return best_box

    def _fallback_tumor_bbox(
        self,
        gray: np.ndarray,
        label: str,
        bx: int,
        by: int,
        bw: int,
        bh: int,
    ) -> tuple[int, int, int, int]:
        h, w = gray.shape[:2]
        normalized_label = str(label).strip().lower().replace(" ", "")

        if normalized_label == "pituitary":
            cx = int(w * 0.50)
            cy = int(h * 0.48)
            half_w = max(18, int(w * 0.08))
            half_h = max(18, int(h * 0.08))
        elif normalized_label == "meningioma":
            cx = int(w * 0.62)
            cy = int(h * 0.34)
            half_w = max(22, int(w * 0.10))
            half_h = max(22, int(h * 0.10))
        else:
            cx = bx + bw // 2
            cy = by + bh // 2
            half_w = max(20, int(w * 0.09))
            half_h = max(20, int(h * 0.09))

        return (
            max(0, cx - half_w),
            max(0, cy - half_h),
            min(w, cx + half_w),
            min(h, cy + half_h),
        )
    def generate_gradcam(self, image_path: Path, output_path: Path, class_id: int | None = None) -> dict[str, Any]:
        """Generate a best-effort Grad-CAM for supported YOLO classification backbones.

        The map represents model activation/attention, not tumour segmentation.
        Unsupported model graphs return a clear unavailable status instead of a fake map.
        """
        try:
            import torch
        except Exception:
            return {"available": False, "reason": "PyTorch is unavailable in this environment."}

        if not self.loaded or self.model is None:
            return {"available": False, "reason": "The classification model is not loaded."}

        frame = cv2.imread(str(image_path))
        if frame is None:
            return {"available": False, "reason": "The MRI image could not be opened."}

        raw_model = getattr(self.model, "model", None)
        if raw_model is None or not hasattr(raw_model, "modules"):
            return {"available": False, "reason": "The loaded model does not expose a compatible PyTorch graph."}

        activations: list[Any] = []
        gradients: list[Any] = []
        handles = []
        target_layer = None
        for module in raw_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        if target_layer is None:
            return {"available": False, "reason": "No convolutional layer was found for Grad-CAM."}

        def capture_activation(_module, _inputs, output):
            if torch.is_tensor(output) and output.ndim == 4:
                activations.append(output)
                output.register_hook(lambda gradient: gradients.append(gradient))

        try:
            handles.append(target_layer.register_forward_hook(capture_activation))
            height, width = 224, 224
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
            try:
                tensor = tensor.to(next(raw_model.parameters()).device)
            except Exception:
                pass
            raw_model.eval()
            raw_model.zero_grad(set_to_none=True)
            with torch.inference_mode(False), torch.enable_grad():
                prediction = raw_model(tensor)
                if isinstance(prediction, (tuple, list)):
                    prediction = next((item for item in prediction if torch.is_tensor(item)), None)
                if not torch.is_tensor(prediction) or prediction.ndim < 2:
                    return {"available": False, "reason": "The model forward pass did not produce class logits."}
                scores = prediction.reshape(prediction.shape[0], -1)
                selected_class = int(class_id) if class_id is not None and 0 <= int(class_id) < scores.shape[1] else int(scores[0].argmax().item())
                scores[0, selected_class].backward()
            if not activations or not gradients:
                return {"available": False, "reason": "This model graph did not expose spatial gradients for Grad-CAM."}
            activation = activations[-1].detach()
            gradient = gradients[-1].detach()
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=frame.shape[:2], mode="bilinear", align_corners=False)[0, 0]
            cam = cam.cpu().numpy()
            cam -= float(cam.min())
            maximum = float(cam.max())
            if maximum <= 1e-8:
                return {"available": False, "reason": "The generated activation map was empty for this prediction."}
            cam = np.uint8(np.clip(cam / maximum * 255.0, 0, 255))
            heat = cv2.applyColorMap(cam, cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(frame, 0.58, heat, 0.42, 0)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), overlay):
                return {"available": False, "reason": "The Grad-CAM image could not be saved."}
            names = getattr(self.model, "names", {})
            try:
                label = names[selected_class] if isinstance(names, dict) else names[selected_class]
            except Exception:
                label = str(selected_class)
            return {"available": True, "class": str(label), "class_id": selected_class, "note": "Grad-CAM indicates model attention; it is not a tumour boundary or clinical segmentation."}
        except Exception as exc:
            return {"available": False, "reason": f"Grad-CAM is unsupported for this model configuration: {exc}"}
        finally:
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            try:
                raw_model.zero_grad(set_to_none=True)
            except Exception:
                pass

