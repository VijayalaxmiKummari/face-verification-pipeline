"""
evaluate_quality.py - measure what the frame clean-up is worth.

For each sample photo, a clean copy is enrolled. Then damaged copies (grain,
darkness) are compared with it twice: once with the raw damaged frame and once
after the clean-up in facepipe/quality.py. A lower hash distance is better, and
anything above the threshold would be a failed verification.

    python3 scripts/evaluate_quality.py
"""

import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from facepipe import quality, template  # noqa: E402
from facepipe.detector import FaceAnalyser  # noqa: E402

KEY, SALT = b"evaluation-key-0123456789abcdef!", b"evaluation-salt!"


def hash_of(analyser, image):
    faces = analyser.detect(image)
    if not faces:
        return None
    return template.biohash(analyser.embed(analyser.align(image, faces[0])), KEY, SALT)


def show(value):
    return "no face " if value is None else "{:.3f}   ".format(value)


def main():
    analyser = FaceAnalyser()
    rng = np.random.default_rng(0)
    print("Hash distance to the clean enrolment (threshold {:.3f}, lower is better)\n".format(
        template.MATCH_THRESHOLD))
    print("{:<10} {:<22} {:<9} {:<9} {}".format("photo", "damage", "raw", "cleaned", "fix applied"))

    for name in ("person_a", "person_b"):
        image = cv2.imread(os.path.join(ROOT, "samples", name + ".jpg"))
        enrolled = hash_of(analyser, image)
        damaged = [("grain, sigma {}".format(s),
                    np.clip(image.astype(np.float32) + rng.normal(0, s, image.shape), 0, 255).astype(np.uint8))
                   for s in (10, 20, 30, 45, 60)]
        damaged += [("dark, {}% light".format(int(g * 100)), (image * g).astype(np.uint8)) for g in (0.25, 0.10)]

        for label, frame in damaged:
            cleaned, report = quality.clean_frame(frame)
            raw_hash, clean_hash = hash_of(analyser, frame), hash_of(analyser, cleaned)
            raw = None if raw_hash is None else template.distance(raw_hash, enrolled)
            fixed = None if clean_hash is None else template.distance(clean_hash, enrolled)
            print("{:<10} {:<22} {:<9} {:<9} {}".format(
                name, label, show(raw), show(fixed), ", ".join(report["fixes"]) or "none"))
        print()


if __name__ == "__main__":
    main()
