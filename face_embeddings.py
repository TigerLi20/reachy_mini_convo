"""Local OpenCV YuNet/SFace detection, alignment, and embeddings."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import urllib.request

import cv2
import numpy as np

MODELS = {
    "face_detection_yunet_2023mar.onnx": ("face_detection_yunet", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec.onnx": ("face_recognition_sface", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}


@dataclass
class RecognitionFace:
    bbox: tuple
    row: np.ndarray


class FaceEmbeddings:
    """Use five-landmark alignment and a pinned 128-dimensional SFace model."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for filename, (folder, digest) in MODELS.items():
            path = directory / filename
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                print(f"Downloading local face model: {filename}", flush=True)
                url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{filename}"
                temporary = path.with_suffix(".download")
                try:
                    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Face model checksum mismatch: {filename}")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        cv2.setNumThreads(1)
        self.detector = cv2.FaceDetectorYN.create(str(directory / "face_detection_yunet_2023mar.onnx"),
                                                "", (320, 240), .8, .3, 100)
        self.encoder = cv2.FaceRecognizerSF.create(str(directory / "face_recognition_sface_2021dec.onnx"), "")

    def detect(self, frame: np.ndarray) -> list[RecognitionFace]:
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, rows = self.detector.detect(frame)
        return [] if rows is None else [RecognitionFace(tuple(row[:4]), row.copy()) for row in rows]

    def prepare(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Preserve face detail while reducing aliasing in the detector image."""
        height, width = frame.shape[:2]
        if width <= 640:
            return frame, (1.0, 1.0)
        resized_height = max(1, round(height * 640 / width))
        small = cv2.resize(frame, (640, resized_height), interpolation=cv2.INTER_AREA)
        return small, (width / small.shape[1], height / small.shape[0])

    def embed(self, frame: np.ndarray, face: RecognitionFace, scale: float | tuple[float, float]) -> np.ndarray:
        row = face.row.copy()
        scale_x, scale_y = scale if isinstance(scale, tuple) else (scale, scale)
        row[:14:2] *= scale_x
        row[1:14:2] *= scale_y
        if min(row[2:4]) < 80:
            raise ValueError("Face too small; come a little closer")
        eyes = row[4:8].reshape(2, 2)
        nose = row[8:10]
        eye_span = float(np.linalg.norm(eyes[1] - eyes[0]))
        if eye_span < 20 or abs(nose[0] - eyes[:, 0].mean()) > eye_span * .45:
            raise ValueError("Face too angled; look toward the camera")
        aligned = self.encoder.alignCrop(frame, row)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        if gray.std() < 12 or cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
            raise ValueError("Face image is blurry or poorly lit")
        embedding = self.encoder.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if embedding.shape != (128,) or not np.isfinite(embedding).all() or norm < 1e-6:
            raise ValueError("Invalid face embedding")
        return embedding / norm
