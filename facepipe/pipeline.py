"""
pipeline.py - the frame-by-frame pipeline.

    frame -> clean up -> find face + landmarks -> align -> sharp enough? -> embedding

It reads frames from a webcam, a video file, a folder of images or one image,
and collects embeddings from the frames that pass the quality checks.
"""

import os

import cv2
import numpy as np

from . import quality
from .detector import draw

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp")
CAMERA_WARMUP_FRAMES = 5          # the first frames from a webcam are usually dark


class SourceError(Exception):
    pass


def frames_from(source):
    """Yield frames from a camera number, a video file, a folder of images or one image."""
    source = str(source)
    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source) if f.lower().endswith(IMAGE_TYPES))
        if not files:
            raise SourceError("No images found in " + source)
        for name in files:
            image = cv2.imread(os.path.join(source, name))
            if image is not None:
                yield image
        return
    if source.lower().endswith(IMAGE_TYPES):
        image = cv2.imread(source)
        if image is None:
            raise SourceError("Could not read the image " + source)
        yield image
        return

    is_camera = source.isdigit()
    capture = cv2.VideoCapture(int(source) if is_camera else source)
    if not capture.isOpened():
        raise SourceError("Could not open the camera. Check that this app is allowed to use it."
                          if is_camera else "Could not open the video " + source)
    try:
        skipped = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if is_camera and skipped < CAMERA_WARMUP_FRAMES:
                skipped += 1
                continue
            yield frame
    finally:
        capture.release()


def collect_embeddings(analyser, frames, wanted=10, max_frames=300, show=False):
    """Run the pipeline until 'wanted' good embeddings are collected or the frames run out.

    Returns (list of embeddings, statistics dictionary).
    """
    stats = {"frames": 0, "no_face": 0, "blurred": 0, "used": 0,
             "denoised": 0, "contrast_boosted": 0}
    embeddings = []

    for frame in frames:
        if stats["frames"] >= max_frames or len(embeddings) >= wanted:
            break
        stats["frames"] += 1

        cleaned, report = quality.clean_frame(frame)
        if "median" in report["fixes"] or "gaussian" in report["fixes"]:
            stats["denoised"] += 1
        if "contrast" in report["fixes"]:
            stats["contrast_boosted"] += 1

        faces = analyser.detect(cleaned)
        status = "no face"
        if not faces:
            stats["no_face"] += 1
        else:
            face = faces[0]                               # the largest face is the one in front
            aligned = analyser.align(cleaned, face)
            if quality.face_is_sharp(aligned):
                embeddings.append(analyser.embed(aligned))
                stats["used"] += 1
                status = "ok {}/{}".format(len(embeddings), wanted)
            else:
                stats["blurred"] += 1
                status = "too blurry"

        if show:
            view = draw(cleaned, faces[0], status) if faces else cleaned
            cv2.imshow("facepipe (press q to stop)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()
    return embeddings, stats


def average(embeddings):
    """Combine several embeddings of one person into one, scaled back to unit length."""
    mean = np.mean(np.stack(embeddings), axis=0)
    return mean / np.linalg.norm(mean)
