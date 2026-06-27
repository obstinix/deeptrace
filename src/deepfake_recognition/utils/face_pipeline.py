"""
src/deepfake_recognition/utils/face_pipeline.py

Face detection, landmark alignment, and crop extraction for DeepTrace.
Uses MediaPipe FaceDetection (bounding box + 6 key-points) for detection
and a similarity transform (rotation + scale only, no shear) for alignment.

Pipeline:
    image → detect_faces() → list[FaceDetection]
                           → align_and_crop()  → list[PIL.Image]  (224×224)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# MediaPipe is an optional dependency; raise a clear error if missing
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    _MP_FACE_DETECTION   = mp.solutions.face_detection
    _MP_DRAWING          = mp.solutions.drawing_utils
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Keypoints:
    """5-point facial landmarks returned by MediaPipe."""
    right_eye:     Tuple[float, float] = (0.0, 0.0)   # normalised (x, y)
    left_eye:      Tuple[float, float] = (0.0, 0.0)
    nose_tip:      Tuple[float, float] = (0.0, 0.0)
    mouth_right:   Tuple[float, float] = (0.0, 0.0)
    mouth_left:    Tuple[float, float] = (0.0, 0.0)


@dataclass
class FaceDetection:
    """
    All data about a single detected face.
    Coordinates are in absolute pixels of the source image.
    """
    # Bounding box
    x1: float
    y1: float
    x2: float
    y2: float

    # Detection confidence
    confidence: float

    # 5-point landmarks in absolute pixel coords
    keypoints: Keypoints = field(default_factory=Keypoints)

    # Face index (0-based) in multi-face images
    face_idx: int = 0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "face_idx":   self.face_idx,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": round(self.x1),
                "y1": round(self.y1),
                "x2": round(self.x2),
                "y2": round(self.y2),
                "width":  round(self.width),
                "height": round(self.height),
            },
            "keypoints": {
                "right_eye":   [round(self.keypoints.right_eye[0]),
                                round(self.keypoints.right_eye[1])],
                "left_eye":    [round(self.keypoints.left_eye[0]),
                                round(self.keypoints.left_eye[1])],
                "nose_tip":    [round(self.keypoints.nose_tip[0]),
                                round(self.keypoints.nose_tip[1])],
                "mouth_right": [round(self.keypoints.mouth_right[0]),
                                round(self.keypoints.mouth_right[1])],
                "mouth_left":  [round(self.keypoints.mouth_left[0]),
                                round(self.keypoints.mouth_left[1])],
            },
        }


# ---------------------------------------------------------------------------
# FacePipeline
# ---------------------------------------------------------------------------

class FacePipeline:
    """
    Stateful face pipeline. Instantiate once per process; reuse across requests.
    MediaPipe's FaceDetection model is loaded on first use and cached.

    Args:
        model_selection:          0 = short-range, 1 = full-range
        min_detection_confidence: minimum score to keep a detection
        margin:                   fractional padding around the bounding box
                                  before cropping. 0.3 = 30% of face size.
        output_size:              final crop size (square), default 224
        align:                    whether to apply similarity-transform alignment
        max_faces:                cap on how many faces to process per image.
                                  None = no limit.
    """

    def __init__(
        self,
        model_selection: int = 1,
        min_detection_confidence: float = 0.5,
        margin: float = 0.30,
        output_size: int = 224,
        align: bool = True,
        max_faces: Optional[int] = None,
    ):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is required for face detection. "
                "Install with: pip install mediapipe"
            )
        self.model_selection          = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.margin                   = margin
        self.output_size              = output_size
        self.align                    = align
        self.max_faces                = max_faces
        self._detector: Optional[object] = None

    # ------------------------------------------------------------------ init

    def _get_detector(self):
        """Lazy-initialise the MediaPipe detector (thread-unsafe — call once)."""
        if self._detector is None:
            self._detector = _MP_FACE_DETECTION.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_detection_confidence,
            )
        return self._detector

    def close(self):
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    # ---------------------------------------------------------------- detect

    def detect(self, image: Image.Image) -> List[FaceDetection]:
        """
        Detect all faces in a PIL image.

        Returns a list of FaceDetection objects sorted by descending area
        (largest face first). Returns [] if no faces are found.
        """
        img_rgb = np.array(image.convert("RGB"))
        h, w    = img_rgb.shape[:2]

        detector = self._get_detector()
        results  = detector.process(img_rgb)

        if not results.detections:
            return []

        faces = []
        for idx, det in enumerate(results.detections):
            if self.max_faces is not None and idx >= self.max_faces:
                break

            bb    = det.location_data.relative_bounding_box
            score = det.score[0] if det.score else 0.0

            # Convert normalised → absolute pixels
            x1 = max(0.0, bb.xmin * w)
            y1 = max(0.0, bb.ymin * h)
            x2 = min(float(w), (bb.xmin + bb.width)  * w)
            y2 = min(float(h), (bb.ymin + bb.height) * h)

            # Extract 6 key-points (MediaPipe returns 6; we use 5)
            kp = det.location_data.relative_keypoints
            def _abs(p):
                return (p.x * w, p.y * h)

            keypoints = Keypoints(
                right_eye   = _abs(kp[0]),
                left_eye    = _abs(kp[1]),
                nose_tip    = _abs(kp[2]),
                mouth_right = _abs(kp[3]),
                mouth_left  = _abs(kp[4]),
            )

            faces.append(FaceDetection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(score),
                keypoints=keypoints,
                face_idx=idx,
            ))

        # Sort by area descending (largest face = most prominent)
        faces.sort(key=lambda f: f.area, reverse=True)

        # Re-index after sort
        for i, f in enumerate(faces):
            f.face_idx = i

        return faces

    # ----------------------------------------------------------------- align

    @staticmethod
    def _get_alignment_transform(
        keypoints: Keypoints,
        output_size: int,
    ) -> np.ndarray:
        """
        Compute a 2×3 similarity transform matrix (rotation + scale + translation)
        that maps the face to a canonical frontal pose.

        Target eye positions are at fixed fractions of the output image,
        matching the standard used by most deepfake datasets.
        """
        # Target eye positions in the output crop
        # Left eye (from the face's perspective) at ~35% from left, 35% from top
        # Right eye at ~65% from left, 35% from top
        target_size = output_size
        tgt_left_eye  = np.array([target_size * 0.35, target_size * 0.35])
        tgt_right_eye = np.array([target_size * 0.65, target_size * 0.35])

        # Source eye positions from landmarks
        # Note: MediaPipe kp[0] = right eye from DETECTOR's perspective = face's left
        src_left_eye  = np.array(keypoints.right_eye)   # face left
        src_right_eye = np.array(keypoints.left_eye)    # face right

        # Compute similarity transform: scale + rotation (no shear)
        src_dir = src_right_eye  - src_left_eye
        tgt_dir = tgt_right_eye  - tgt_left_eye

        src_angle  = math.atan2(src_dir[1], src_dir[0])
        tgt_angle  = math.atan2(tgt_dir[1], tgt_dir[0])
        angle_diff = tgt_angle - src_angle

        src_dist   = float(np.linalg.norm(src_dir))
        tgt_dist   = float(np.linalg.norm(tgt_dir))
        scale      = tgt_dist / (src_dist + 1e-6)

        # Build rotation matrix around the midpoint between eyes
        cos_a, sin_a = math.cos(angle_diff), math.sin(angle_diff)
        src_mid = (src_left_eye + src_right_eye) / 2.0
        tgt_mid = (tgt_left_eye + tgt_right_eye) / 2.0

        # M such that: tgt = scale * R * (src - src_mid) + tgt_mid
        M = np.array([
            [scale * cos_a, -scale * sin_a,
             tgt_mid[0] - scale * (cos_a * src_mid[0] - sin_a * src_mid[1])],
            [scale * sin_a,  scale * cos_a,
             tgt_mid[1] - scale * (sin_a * src_mid[0] + cos_a * src_mid[1])],
        ], dtype=np.float32)

        return M

    # ------------------------------------------------------------------ crop

    def crop(
        self,
        image: Image.Image,
        face: FaceDetection,
    ) -> Image.Image:
        """
        Crop and align a single face from a PIL image.
        Returns a PIL Image of size (output_size × output_size).
        """
        img_np = np.array(image.convert("RGB"))
        h, w   = img_np.shape[:2]

        if self.align and face.keypoints is not None:
            # Alignment path: warp the full image, then centre-crop the face
            M = self._get_alignment_transform(face.keypoints, self.output_size)
            warped = cv2.warpAffine(
                img_np, M,
                (self.output_size, self.output_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return Image.fromarray(warped)

        else:
            # No alignment: simple bbox crop with margin + resize
            face_w = face.x2 - face.x1
            face_h = face.y2 - face.y1
            pad_x  = face_w * self.margin
            pad_y  = face_h * self.margin

            cx1 = max(0, int(face.x1 - pad_x))
            cy1 = max(0, int(face.y1 - pad_y))
            cx2 = min(w, int(face.x2 + pad_x))
            cy2 = min(h, int(face.y2 + pad_y))

            cropped = img_np[cy1:cy2, cx1:cx2]

            # Pad to square
            ch, cw = cropped.shape[:2]
            side   = max(ch, cw)
            canvas = np.zeros((side, side, 3), dtype=np.uint8)
            yo     = (side - ch) // 2
            xo     = (side - cw) // 2
            canvas[yo:yo+ch, xo:xo+cw] = cropped

            out = cv2.resize(canvas, (self.output_size, self.output_size),
                             interpolation=cv2.INTER_LINEAR)
            return Image.fromarray(out)

    # --------------------------------------------------------------- process

    def process(
        self,
        image: Image.Image,
    ) -> Tuple[List[FaceDetection], List[Image.Image]]:
        """
        Full pipeline: detect all faces, crop and align each one.

        Returns:
            (detections, crops)
            detections: list of FaceDetection (may be empty)
            crops:      parallel list of 224×224 PIL images
                        Empty if no faces found.
        """
        detections = self.detect(image)
        crops      = [self.crop(image, det) for det in detections]
        return detections, crops

    # ----------------------------------------------------------- draw_boxes

    @staticmethod
    def draw_boxes(
        image: Image.Image,
        detections: List[FaceDetection],
        color: Tuple[int, int, int] = (0, 212, 255),   # DeepTrace cyan
        thickness: int = 2,
        draw_keypoints: bool = True,
    ) -> Image.Image:
        """
        Draw bounding boxes (and optionally keypoints) on a PIL image.
        Returns a new PIL image — does not modify the original.
        """
        img_np = np.array(image.convert("RGB"))

        for det in detections:
            cv2.rectangle(
                img_np,
                (int(det.x1), int(det.y1)),
                (int(det.x2), int(det.y2)),
                color, thickness,
            )
            # Face index label
            cv2.putText(
                img_np,
                f"#{det.face_idx}  {det.confidence:.2f}",
                (int(det.x1), max(0, int(det.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
            if draw_keypoints:
                for kp_name in ("right_eye", "left_eye", "nose_tip",
                                "mouth_right", "mouth_left"):
                    pt = getattr(det.keypoints, kp_name)
                    cv2.circle(img_np, (int(pt[0]), int(pt[1])),
                               3, color, -1)

        return Image.fromarray(img_np)
