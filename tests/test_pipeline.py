"""
End-to-end tests with the real face models and the two sample photographs.

They are skipped until the model files have been downloaded:
    python3 verify.py download-models
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from facepipe import models, pipeline, template  # noqa: E402

PERSON_A = os.path.join(ROOT, "samples", "person_a.jpg")
PERSON_B = os.path.join(ROOT, "samples", "person_b.jpg")
KEY = b"k" * 32

try:
    for _name in models.MODELS:
        models.model_path(_name, download=False)
    HAVE_MODELS = True
except FileNotFoundError:
    HAVE_MODELS = False


def add_noise(image, sigma, seed=0):
    noise = np.random.default_rng(seed).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@unittest.skipUnless(HAVE_MODELS, "model files not downloaded (run: python3 verify.py download-models)")
class ModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from facepipe.detector import FaceAnalyser

        cls.analyser = FaceAnalyser(download=False)
        cls.a = cv2.imread(PERSON_A)
        cls.b = cv2.imread(PERSON_B)

    def embedding(self, image):
        embeddings, _ = pipeline.collect_embeddings(self.analyser, [image], wanted=1)
        return embeddings[0] if embeddings else None


class DetectionTests(ModelTestCase):
    def test_one_face_with_five_landmarks(self):
        for image in (self.a, self.b):
            faces = self.analyser.detect(image)
            self.assertEqual(len(faces), 1)
            self.assertEqual(len(faces[0].landmarks), 5)
            self.assertGreater(faces[0].confidence, 0.8)

    def test_landmarks_are_inside_the_box_and_in_the_right_order(self):
        for image in (self.a, self.b):
            face = self.analyser.detect(image)[0]
            x, y, w, h = face.box
            for px, py in face.landmarks:
                self.assertTrue(x <= px <= x + w and y <= py <= y + h)
            right_eye, left_eye, nose, right_mouth, left_mouth = face.landmarks
            self.assertLess(right_eye[0], left_eye[0])               # eyes left to right in the image
            self.assertLess(max(right_eye[1], left_eye[1]), nose[1])    # eyes above the nose
            self.assertLess(nose[1], min(right_mouth[1], left_mouth[1]))   # nose above the mouth

    def test_no_face_in_a_blank_frame(self):
        self.assertEqual(self.analyser.detect(np.full((480, 640, 3), 127, np.uint8)), [])

    def test_detector_copes_with_a_change_of_frame_size(self):
        self.assertEqual(len(self.analyser.detect(self.a)), 1)
        self.assertEqual(len(self.analyser.detect(cv2.resize(self.a, (320, 376)))), 1)

    def test_aligned_face_is_112_pixels_square(self):
        aligned = self.analyser.align(self.a, self.analyser.detect(self.a)[0])
        self.assertEqual(aligned.shape, (112, 112, 3))

    def test_embedding_is_128_numbers_of_unit_length(self):
        embedding = self.embedding(self.a)
        self.assertEqual(embedding.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)


class VerificationTests(ModelTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = template.TemplateStore(self.folder.name, KEY)
        self.store.enrol("person_a", self.embedding(self.a))

    def test_same_photo_matches(self):
        self.assertLess(self.store.compare("person_a", self.embedding(self.a)), 0.01)

    def test_different_person_does_not_match(self):
        self.assertGreater(self.store.compare("person_a", self.embedding(self.b)), template.MATCH_THRESHOLD)

    def test_same_person_still_matches_in_poor_conditions(self):
        conditions = {
            "grainy": add_noise(self.a, 30),
            "very grainy": add_noise(self.a, 50),
            "dark": (self.a * 0.15).astype(np.uint8),
            "dark and grainy": add_noise((self.a * 0.3).astype(np.uint8), 20),
            "smaller": cv2.resize(self.a, None, fx=0.5, fy=0.5),
            "mirrored": cv2.flip(self.a, 1),
            "slightly blurred": cv2.GaussianBlur(self.a, (7, 7), 0),
        }
        for label, frame in conditions.items():
            embedding = self.embedding(frame)
            self.assertIsNotNone(embedding, label)
            self.assertLess(self.store.compare("person_a", embedding), template.MATCH_THRESHOLD, label)

    def test_cleaning_brings_a_grainy_frame_closer_to_the_template(self):
        for sigma in (30, 45):
            noisy = add_noise(self.a, sigma)
            face = self.analyser.detect(noisy)[0]
            raw = self.analyser.embed(self.analyser.align(noisy, face))     # no clean-up
            cleaned = self.embedding(noisy)                                 # through the pipeline
            self.assertLess(self.store.compare("person_a", cleaned),
                            self.store.compare("person_a", raw), sigma)


class PipelineTests(ModelTestCase):
    def test_statistics_count_every_kind_of_frame(self):
        blank = np.full((480, 640, 3), 127, np.uint8)
        blurred = cv2.GaussianBlur(self.b, (31, 31), 0)
        frames = [self.a, blank, add_noise(self.a, 30), blurred, (self.a * 0.15).astype(np.uint8)]
        embeddings, stats = pipeline.collect_embeddings(self.analyser, frames, wanted=10)
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(stats["frames"], 5)
        self.assertEqual(stats["used"], 3)
        self.assertEqual(stats["no_face"], 1)
        self.assertEqual(stats["blurred"], 1)
        self.assertEqual(stats["denoised"], 1)
        self.assertEqual(stats["contrast_boosted"], 1)

    def test_stops_when_enough_frames_are_collected(self):
        embeddings, stats = pipeline.collect_embeddings(self.analyser, [self.a] * 20, wanted=4)
        self.assertEqual((len(embeddings), stats["frames"]), (4, 4))

    def test_average_is_unit_length(self):
        embeddings, _ = pipeline.collect_embeddings(self.analyser, [self.a, add_noise(self.a, 20)], wanted=2)
        self.assertAlmostEqual(float(np.linalg.norm(pipeline.average(embeddings))), 1.0, places=5)

    def test_frames_from_an_image_and_a_folder(self):
        self.assertEqual(len(list(pipeline.frames_from(PERSON_A))), 1)
        self.assertEqual(len(list(pipeline.frames_from(os.path.join(ROOT, "samples")))), 2)

    def test_frames_from_a_video_file(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        path = os.path.join(folder, "clip.avi")
        height, width = self.a.shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
        if not writer.isOpened():
            self.skipTest("this OpenCV build cannot write video")
        for _ in range(6):
            writer.write(self.a)
        writer.release()
        embeddings, stats = pipeline.collect_embeddings(self.analyser, pipeline.frames_from(path), wanted=5)
        self.assertEqual((len(embeddings), stats["frames"]), (5, 5))

    def test_bad_sources(self):
        with self.assertRaises(pipeline.SourceError):
            list(pipeline.frames_from("no-such-video.mp4"))
        with self.assertRaises(pipeline.SourceError):
            list(pipeline.frames_from(tempfile.mkdtemp()))


@unittest.skipUnless(HAVE_MODELS, "model files not downloaded (run: python3 verify.py download-models)")
class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder)

    def run_cli(self, *args):
        command = [sys.executable, os.path.join(ROOT, "verify.py"),
                   "--store", os.path.join(self.folder, "templates"),
                   "--key-file", os.path.join(self.folder, "facepipe.key")] + list(args)
        return subprocess.run(command, capture_output=True, text=True, timeout=120)

    def test_enrol_verify_list_delete(self):
        result = self.run_cli("enrol", "grace", "--source", PERSON_A)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("grace", self.run_cli("list").stdout)

        same = self.run_cli("verify", "grace", "--source", PERSON_A)
        self.assertEqual(same.returncode, 0, same.stdout + same.stderr)
        self.assertIn("MATCH", same.stdout)

        other = self.run_cli("verify", "grace", "--source", PERSON_B)
        self.assertEqual(other.returncode, 1)
        self.assertIn("NO MATCH", other.stdout)

        self.assertEqual(self.run_cli("delete", "grace").returncode, 0)
        self.assertNotEqual(self.run_cli("verify", "grace", "--source", PERSON_A).returncode, 0)

    def test_nothing_but_the_hash_and_key_is_written(self):
        self.run_cli("enrol", "grace", "--source", PERSON_A)
        written = sorted(os.path.relpath(os.path.join(folder, name), self.folder)
                         for folder, _, names in os.walk(self.folder) for name in names)
        self.assertEqual(written, ["facepipe.key", os.path.join("templates", "grace.json")])

    def test_bad_name_is_refused(self):
        result = self.run_cli("enrol", "../evil", "--source", PERSON_A)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(self.folder, "evil.json")))

    def test_landmarks_command_writes_an_image(self):
        output = os.path.join(self.folder, "out.jpg")
        result = self.run_cli("landmarks", "--source", PERSON_A, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nose tip", result.stdout)
        self.assertIsNotNone(cv2.imread(output))


if __name__ == "__main__":
    unittest.main()
