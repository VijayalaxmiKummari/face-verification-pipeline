"""
template.py - protect the stored biometric data.

A face embedding is biometric data. If a database of raw embeddings leaked, the
faces could be matched against other systems, and unlike a password a face
cannot be changed. So this pipeline never stores an embedding or an image.

An ordinary hash such as SHA-256 cannot be used here: two photos of the same
face never give exactly the same embedding, and a tiny change in the input
changes a SHA-256 hash completely. Instead the embedding is turned into a
"BioHash", a keyed locality-sensitive hash:

  1. A secret key and a random per-person salt are stretched (with SHAKE-256)
     into a table of +1/-1 values.
  2. The embedding is projected through that table and only the SIGN of each
     result is kept, giving 2048 bits.
  3. Two hashes of the same face differ in few bits; hashes of different faces
     differ in about half. Verification counts the differing bits.

What this gives:
  * The stored bits are useless without the secret key.
  * If a template leaks it can be cancelled: enrol again with a new salt and
    the old template no longer matches anything.
  * The same face enrolled in two systems (different keys) cannot be linked.

Each template also carries an HMAC, so a file that has been edited is rejected.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime, timezone

import numpy as np

N_BITS = 2048
EMBEDDING_SIZE = 128
SALT_BYTES = 16
KEY_BYTES = 32
# SFace's published decision threshold is a cosine similarity of 0.363. A sign
# projection turns an angle into a fraction of differing bits (angle / pi), so
# the same threshold in hash form is acos(0.363) / pi = 0.382.
MATCH_THRESHOLD = 0.382
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
VERSION = 1


class TemplateError(Exception):
    pass


def load_or_create_key(path):
    """Read the secret key, or create one on first use (readable only by the owner)."""
    env_key = os.environ.get("FACEPIPE_KEY")
    if env_key:
        try:
            key = bytes.fromhex(env_key)
        except ValueError:
            raise TemplateError("FACEPIPE_KEY must be a hexadecimal string") from None
        if len(key) < KEY_BYTES:
            raise TemplateError("FACEPIPE_KEY must be at least {} bytes".format(KEY_BYTES))
        return key
    if os.path.exists(path):
        with open(path, "rb") as handle:
            key = handle.read()
        if len(key) < KEY_BYTES:
            raise TemplateError("The key file is too short: " + path)
        return key
    key = secrets.token_bytes(KEY_BYTES)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(key)
    return key


def projection_table(key, salt, n_bits=N_BITS, size=EMBEDDING_SIZE):
    """Stretch key + salt into an n_bits x size table of +1/-1 values.

    hashlib is used, not NumPy's random generator, so the table is identical on
    every machine and every library version.
    """
    stream = hashlib.shake_256(b"facepipe-biohash-v1" + key + salt).digest(n_bits * size // 8)
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))
    return (bits.astype(np.float32) * 2 - 1).reshape(n_bits, size)


def biohash(embedding, key, salt):
    """Turn an embedding into N_BITS bits, returned as bytes."""
    embedding = np.asarray(embedding, dtype=np.float32).flatten()
    if embedding.shape != (EMBEDDING_SIZE,):
        raise TemplateError("expected an embedding of {} numbers".format(EMBEDDING_SIZE))
    signs = projection_table(key, salt) @ embedding > 0
    return np.packbits(signs).tobytes()


def distance(hash_a, hash_b):
    """Fraction of bits that differ: about 0.0 for the same input, about 0.5 for unrelated ones."""
    a = np.frombuffer(hash_a, dtype=np.uint8)
    b = np.frombuffer(hash_b, dtype=np.uint8)
    if a.shape != b.shape:
        raise TemplateError("hashes have different lengths")
    return float(np.unpackbits(a ^ b).sum()) / (a.size * 8)


def _mac(key, name, salt, bits):
    message = b"|".join([str(VERSION).encode(), name.encode(), salt, bits])
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def check_name(name):
    """Names become file names, so only plain characters are allowed (no ../ tricks)."""
    if not NAME_PATTERN.match(name or ""):
        raise TemplateError("Name must be 1-32 letters, digits, hyphens or underscores.")
    return name


class TemplateStore:
    """Saves and loads protected templates as small JSON files."""

    def __init__(self, folder, key):
        self.folder = folder
        self.key = key
        os.makedirs(folder, mode=0o700, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.folder, check_name(name) + ".json")

    def enrol(self, name, embedding, frames=1):
        """Hash an embedding with a fresh salt and store only the hash."""
        path = self._path(name)                             # checks the name before anything else
        salt = secrets.token_bytes(SALT_BYTES)
        bits = biohash(embedding, self.key, salt)
        record = {"version": VERSION, "name": name, "salt": salt.hex(), "hash": bits.hex(),
                  "bits": N_BITS, "frames": frames,
                  "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "mac": _mac(self.key, name, salt, bits)}
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        return record

    def load(self, name):
        path = self._path(name)
        if not os.path.exists(path):
            raise TemplateError("Nobody is enrolled as '{}'.".format(name))
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            salt, bits = bytes.fromhex(record["salt"]), bytes.fromhex(record["hash"])
            stored_mac = record["mac"]
        except (ValueError, KeyError, TypeError):
            raise TemplateError("The template for '{}' is damaged.".format(name)) from None
        if not hmac.compare_digest(stored_mac, _mac(self.key, name, salt, bits)):
            raise TemplateError("The template for '{}' failed its integrity check "
                                "(edited file, or a different key).".format(name))
        return salt, bits

    def compare(self, name, embedding):
        """Return the hash distance between a fresh embedding and the enrolled template."""
        salt, bits = self.load(name)
        return distance(biohash(embedding, self.key, salt), bits)

    def names(self):
        return sorted(f[:-5] for f in os.listdir(self.folder) if f.endswith(".json"))

    def delete(self, name):
        path = self._path(name)
        if not os.path.exists(path):
            raise TemplateError("Nobody is enrolled as '{}'.".format(name))
        os.remove(path)
