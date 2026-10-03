"""Tests for the frame quality checks and clean-up. These need OpenCV and NumPy but no model files."""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from facepipe import quality as q  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "person_a.jpg")


def add_noise(image, sigma, seed=0):
    noise = np.random.default_rng(seed).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imread(SAMPLE)

    def test_clean_image_is_left_alone(self):
        cleaned, report = q.clean_frame(self.image)
        self.assertEqual(report["fixes"], [])
        self.assertTrue(np.array_equal(cleaned, self.image))

    def test_noise_estimate_rises_with_noise(self):
        levels = [q.noise_level(add_noise(self.image, sigma)) for sigma in (0, 10, 20, 40)]
        self.assertEqual(levels, sorted(levels))
        self.assertLess(levels[0], q.NOISE_MILD)
        self.assertGreater(levels[-1], q.NOISE_HEAVY)

    def test_flat_image_has_no_noise(self):
        self.assertAlmostEqual(q.noise_level(np.full((120, 160, 3), 128, np.uint8)), 0.0)

    def test_light_grain_gets_the_median_filter(self):
        cleaned, report = q.clean_frame(add_noise(self.image, 15))
        self.assertEqual(report["fixes"], ["median"])
        self.assertLess(q.noise_level(cleaned), report["noise"])

    def test_heavy_grain_gets_the_stronger_filter(self):
        cleaned, report = q.clean_frame(add_noise(self.image, 45))
        self.assertEqual(report["fixes"], ["gaussian"])
        self.assertLess(q.noise_level(cleaned), report["noise"] / 3)

    def test_cleaning_moves_the_frame_back_towards_the_original(self):
        noisy = add_noise(self.image, 45)
        cleaned, _ = q.clean_frame(noisy)
        error_before = np.abs(noisy.astype(np.float32) - self.image).mean()
        error_after = np.abs(cleaned.astype(np.float32) - self.image).mean()
        self.assertLess(error_after, error_before)

    def test_dark_frame_is_brightened(self):
        dark = (self.image * 0.15).astype(np.uint8)
        cleaned, report = q.clean_frame(dark)
        self.assertIn("contrast", report["fixes"])
        self.assertGreater(q.brightness(cleaned), q.brightness(dark))

    def test_dark_and_grainy_gets_both_fixes(self):
        frame = add_noise((self.image * 0.2).astype(np.uint8), 25)
        _, report = q.clean_frame(frame)
        self.assertIn("contrast", report["fixes"])
        self.assertTrue({"median", "gaussian"} & set(report["fixes"]))

    def test_sharpness_drops_with_blur(self):
        scores = [q.sharpness(self.image)] + [q.sharpness(cv2.GaussianBlur(self.image, (k, k), 0)) for k in (5, 15, 31)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_blurred_crop_is_rejected(self):
        crop = cv2.resize(self.image[100:340, 170:360], (112, 112))
        self.assertTrue(q.face_is_sharp(crop))
        self.assertFalse(q.face_is_sharp(cv2.GaussianBlur(crop, (21, 21), 0)))

    def test_grey_images_are_accepted(self):
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.assertGreater(q.sharpness(grey), 0)
        self.assertGreater(q.brightness(grey), 0)


if __name__ == "__main__":
    unittest.main()
