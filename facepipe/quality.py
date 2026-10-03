"""
quality.py - measure how good a video frame is, and clean up the poor ones.

Webcam frames are often grainy (low light pushes the sensor gain up), too dark,
or blurred by movement. Grain and darkness can be repaired, so those frames are
cleaned and kept. Blur cannot be repaired, so blurred faces are thrown away.

The thresholds below were chosen by adding known amounts of noise, darkness and
blur to the sample images and measuring how far the face embedding moved
(see scripts/evaluate_quality.py).
"""

import cv2
import numpy as np

NOISE_MILD = 8.0          # estimated noise level above which a frame is filtered
NOISE_HEAVY = 15.0        # above this a stronger filter is used
DARK_BELOW = 60.0         # mean brightness (0-255) below which contrast is boosted
MIN_FACE_SHARPNESS = 40.0  # an aligned face crop less sharp than this is rejected

# Immerkaer's kernel: it cancels out image structure and leaves mostly noise.
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def noise_level(image):
    """Estimate the standard deviation of the noise in an image (Immerkaer, 1996)."""
    gray = to_gray(image).astype(np.float32)
    height, width = gray.shape
    response = cv2.filter2D(gray, -1, _NOISE_KERNEL)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(response).sum() / (6 * (width - 2) * (height - 2)))


def brightness(image):
    return float(to_gray(image).mean())


def sharpness(image):
    """Variance of the Laplacian: high for crisp edges, low for a blurred image."""
    return float(cv2.Laplacian(to_gray(image), cv2.CV_64F).var())


def boost_contrast(image):
    """Brighten a dark frame with CLAHE on the lightness channel, leaving colour alone."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clean_frame(frame):
    """Repair a frame where possible. Returns (cleaned frame, report dictionary).

    The report records what was measured and which fixes were applied, so the
    caller can log it or show it on screen.
    """
    report = {"noise": noise_level(frame), "brightness": brightness(frame), "fixes": []}
    cleaned = frame

    if report["noise"] > NOISE_HEAVY:
        cleaned = cv2.GaussianBlur(cleaned, (5, 5), 0)      # heavy grain: smooth harder
        report["fixes"].append("gaussian")
    elif report["noise"] > NOISE_MILD:
        cleaned = cv2.medianBlur(cleaned, 3)                # light grain: median keeps edges
        report["fixes"].append("median")

    if report["brightness"] < DARK_BELOW:
        cleaned = boost_contrast(cleaned)
        report["fixes"].append("contrast")

    return cleaned, report


def face_is_sharp(aligned_face, threshold=MIN_FACE_SHARPNESS):
    """Check an aligned 112x112 face crop. Measuring the crop, not the whole frame,
    means the answer does not depend on the camera resolution or the background."""
    return sharpness(aligned_face) >= threshold
