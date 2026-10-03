"""
models.py - fetch the two neural network files the pipeline needs.

  YuNet  finds faces and five landmarks (eyes, nose tip, mouth corners)
  SFace  turns an aligned face into a 128-number embedding

Both come from the OpenCV Model Zoo. They are too large to keep in the
repository, so they are downloaded on first use. Each file is checked against
a known SHA-256 hash, so a corrupted or tampered download is never loaded.
"""

import hashlib
import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
BASE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"

MODELS = {
    "detector": {
        "file": "face_detection_yunet_2023mar.onnx",
        "url": BASE_URL + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "embedder": {
        "file": "face_recognition_sface_2021dec.onnx",
        "url": BASE_URL + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_path(name, download=True):
    """Return the path of a model file, downloading and verifying it if needed."""
    info = MODELS[name]
    path = os.path.join(MODEL_DIR, info["file"])
    if os.path.exists(path) and sha256_of(path) == info["sha256"]:
        return path
    if not download:
        raise FileNotFoundError("{} is missing. Run: python3 verify.py download-models".format(info["file"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[*] Downloading {} ...".format(info["file"]))
    partial = path + ".part"
    urllib.request.urlretrieve(info["url"], partial)
    found = sha256_of(partial)
    if found != info["sha256"]:
        os.remove(partial)
        raise RuntimeError("Checksum mismatch for {}: the download is corrupt or has been altered.".format(info["file"]))
    os.replace(partial, path)
    print("[✓] {} verified".format(info["file"]))
    return path


def download_all():
    return {name: model_path(name) for name in MODELS}
