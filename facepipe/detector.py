"""
detector.py - find faces and their landmarks with YuNet, and embed them with SFace.

Both networks run through OpenCV's own DNN module, so nothing but OpenCV and
NumPy is needed.
"""

import cv2
import numpy as np

from . import models

LANDMARK_NAMES = ("right eye", "left eye", "nose tip", "right mouth corner", "left mouth corner")

try:                                    # keep OpenCV's internal warnings out of the output
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


class Face:
    """One detected face: bounding box, five landmarks and the detector's confidence."""

    def __init__(self, row):
        self.row = np.asarray(row, dtype=np.float32)      # the raw YuNet output, needed for alignment
        self.box = tuple(int(round(v)) for v in row[0:4])                 # x, y, width, height
        self.landmarks = [(float(row[i]), float(row[i + 1])) for i in range(4, 14, 2)]
        self.confidence = float(row[14])

    @property
    def area(self):
        return self.box[2] * self.box[3]


class FaceAnalyser:
    def __init__(self, min_confidence=0.8, download=True):
        self._detector_path = models.model_path("detector", download)
        self._embedder = cv2.FaceRecognizerSF.create(models.model_path("embedder", download), "")
        self._min_confidence = min_confidence
        self._detector = None
        self._size = None

    def detect(self, image):
        """Return every face in the image, largest first."""
        height, width = image.shape[:2]
        if self._size != (width, height):               # the detector is built for one frame size
            self._detector = cv2.FaceDetectorYN.create(
                self._detector_path, "", (width, height), self._min_confidence, 0.3, 5000)
            self._size = (width, height)
        _, rows = self._detector.detect(image)
        faces = [Face(row) for row in rows] if rows is not None else []
        return sorted(faces, key=lambda face: face.area, reverse=True)

    def align(self, image, face):
        """Rotate and crop the face to a standard 112x112 pose using its landmarks."""
        return self._embedder.alignCrop(image, face.row)

    def embed(self, aligned_face):
        """Return the 128-number embedding of an aligned face, scaled to unit length."""
        vector = self._embedder.feature(aligned_face).flatten().astype(np.float32)
        return vector / np.linalg.norm(vector)


def draw(image, face, label=None, colour=(0, 200, 0)):
    """Draw the box, landmarks and an optional label on a copy of the image."""
    canvas = image.copy()
    x, y, w, h = face.box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
    for px, py in face.landmarks:
        cv2.circle(canvas, (int(round(px)), int(round(py))), 4, (0, 215, 255), -1)
    if label:
        cv2.putText(canvas, label, (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return canvas
