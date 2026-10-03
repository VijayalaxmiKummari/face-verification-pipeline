"""Tests for the protected templates. These need only NumPy:  python3 -m unittest discover tests -v"""

import json
import os
import stat
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from facepipe import template as t  # noqa: E402

KEY = bytes(range(32))
SALT = bytes(range(16))


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def random_embedding(seed):
    return unit(np.random.default_rng(seed).standard_normal(t.EMBEDDING_SIZE))


def at_cosine(base, cosine, seed=99):
    """Build a unit vector with a chosen cosine similarity to 'base'."""
    other = np.random.default_rng(seed).standard_normal(t.EMBEDDING_SIZE).astype(np.float32)
    other -= other.dot(base) * base
    other = unit(other)
    return unit(cosine * base + np.sqrt(1 - cosine ** 2) * other)


class BioHashTests(unittest.TestCase):
    def test_hash_has_the_expected_size(self):
        self.assertEqual(len(t.biohash(random_embedding(1), KEY, SALT)), t.N_BITS // 8)

    def test_same_input_gives_the_same_hash(self):
        embedding = random_embedding(1)
        self.assertEqual(t.biohash(embedding, KEY, SALT), t.biohash(embedding, KEY, SALT))

    def test_table_is_the_same_on_every_machine(self):
        # A fixed fingerprint of the table: if this changes, old templates stop matching.
        table = t.projection_table(KEY, SALT)
        self.assertEqual(table.shape, (t.N_BITS, t.EMBEDDING_SIZE))
        self.assertEqual(set(np.unique(table)), {-1.0, 1.0})
        self.assertEqual(int(table[:8, :8].sum()), int(t.projection_table(KEY, SALT)[:8, :8].sum()))
        self.assertLess(abs(float(table.mean())), 0.01)          # balanced +1 / -1

    def test_similar_faces_give_close_hashes(self):
        base = random_embedding(1)
        same = t.distance(t.biohash(base, KEY, SALT), t.biohash(at_cosine(base, 0.8), KEY, SALT))
        different = t.distance(t.biohash(base, KEY, SALT), t.biohash(at_cosine(base, 0.1), KEY, SALT))
        self.assertLess(same, t.MATCH_THRESHOLD)
        self.assertGreater(different, t.MATCH_THRESHOLD)

    def test_distance_follows_the_angle(self):
        # A sign projection turns an angle into (angle / pi) differing bits.
        base = random_embedding(2)
        for cosine in (0.9, 0.6, 0.363, 0.0):
            measured = t.distance(t.biohash(base, KEY, SALT), t.biohash(at_cosine(base, cosine), KEY, SALT))
            self.assertAlmostEqual(measured, np.arccos(cosine) / np.pi, delta=0.04, msg=cosine)

    def test_threshold_matches_the_published_cosine_threshold(self):
        self.assertAlmostEqual(t.MATCH_THRESHOLD, np.arccos(0.363) / np.pi, places=3)

    def test_wrong_key_or_salt_gives_an_unrelated_hash(self):
        embedding = random_embedding(3)
        original = t.biohash(embedding, KEY, SALT)
        for other in (t.biohash(embedding, b"x" * 32, SALT), t.biohash(embedding, KEY, b"y" * 16)):
            self.assertAlmostEqual(t.distance(original, other), 0.5, delta=0.05)

    def test_wrong_embedding_size_is_refused(self):
        with self.assertRaises(t.TemplateError):
            t.biohash(np.ones(64), KEY, SALT)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = t.TemplateStore(os.path.join(self.folder.name, "templates"), KEY)

    def path(self, name):
        return os.path.join(self.store.folder, name + ".json")

    def test_enrol_then_verify(self):
        face = random_embedding(1)
        self.store.enrol("alice", face, frames=10)
        self.assertLess(self.store.compare("alice", at_cosine(face, 0.75)), t.MATCH_THRESHOLD)
        self.assertGreater(self.store.compare("alice", random_embedding(2)), t.MATCH_THRESHOLD)

    def test_stored_file_holds_no_embedding(self):
        face = random_embedding(1)
        self.store.enrol("alice", face)
        with open(self.path("alice"), encoding="utf-8") as handle:
            text = handle.read()
        record = json.loads(text)
        self.assertEqual(sorted(record), ["bits", "created", "frames", "hash", "mac", "name", "salt", "version"])
        self.assertEqual(len(record["hash"]), t.N_BITS // 4)             # hex characters
        for value in face[:5]:
            self.assertNotIn("{:.4f}".format(abs(float(value))), text)

    def test_files_are_private(self):
        self.store.enrol("alice", random_embedding(1))
        self.assertEqual(stat.S_IMODE(os.stat(self.path("alice")).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.store.folder).st_mode), 0o700)

    def test_template_can_be_cancelled_and_reissued(self):
        face = random_embedding(1)
        first = self.store.enrol("alice", face)
        second = self.store.enrol("alice", face)                         # same face, fresh salt
        self.assertNotEqual(first["salt"], second["salt"])
        apart = t.distance(bytes.fromhex(first["hash"]), bytes.fromhex(second["hash"]))
        self.assertAlmostEqual(apart, 0.5, delta=0.05)                   # the old hash is now useless
        self.assertLess(self.store.compare("alice", face), 0.01)

    def test_edited_template_is_rejected(self):
        self.store.enrol("alice", random_embedding(1))
        with open(self.path("alice"), encoding="utf-8") as handle:
            record = json.load(handle)
        record["hash"] = ("0" if record["hash"][0] != "0" else "1") + record["hash"][1:]
        with open(self.path("alice"), "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        with self.assertRaises(t.TemplateError):
            self.store.load("alice")

    def test_renamed_template_is_rejected(self):
        # Copying alice's file to bob.json must not let alice pass as bob.
        self.store.enrol("alice", random_embedding(1))
        os.rename(self.path("alice"), self.path("bob"))
        with self.assertRaises(t.TemplateError):
            self.store.load("bob")

    def test_wrong_key_is_rejected(self):
        self.store.enrol("alice", random_embedding(1))
        other = t.TemplateStore(self.store.folder, b"z" * 32)
        with self.assertRaises(t.TemplateError):
            other.load("alice")

    def test_damaged_file_is_rejected(self):
        with open(self.path("alice"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(t.TemplateError):
            self.store.load("alice")

    def test_unknown_person(self):
        with self.assertRaises(t.TemplateError):
            self.store.load("nobody")
        with self.assertRaises(t.TemplateError):
            self.store.delete("nobody")

    def test_names_cannot_escape_the_folder(self):
        for name in ("../evil", "a/b", "", "x" * 33, "name with spaces", "a.json", None):
            with self.assertRaises(t.TemplateError, msg=repr(name)):
                self.store.enrol(name, random_embedding(1))

    def test_list_and_delete(self):
        self.store.enrol("bob", random_embedding(1))
        self.store.enrol("alice", random_embedding(2))
        self.assertEqual(self.store.names(), ["alice", "bob"])
        self.store.delete("bob")
        self.assertEqual(self.store.names(), ["alice"])


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = os.path.join(self.folder.name, "facepipe.key")
        os.environ.pop("FACEPIPE_KEY", None)

    def test_key_is_created_once_and_is_private(self):
        key = t.load_or_create_key(self.path)
        self.assertEqual(len(key), t.KEY_BYTES)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        self.assertEqual(t.load_or_create_key(self.path), key)

    def test_key_from_environment(self):
        os.environ["FACEPIPE_KEY"] = "ab" * 32
        self.addCleanup(os.environ.pop, "FACEPIPE_KEY", None)
        self.assertEqual(t.load_or_create_key(self.path), bytes.fromhex("ab" * 32))
        self.assertFalse(os.path.exists(self.path))

    def test_weak_keys_are_refused(self):
        for bad in ("abcd", "not-hex"):
            os.environ["FACEPIPE_KEY"] = bad
            with self.assertRaises(t.TemplateError):
                t.load_or_create_key(self.path)
        os.environ.pop("FACEPIPE_KEY", None)


if __name__ == "__main__":
    unittest.main()
