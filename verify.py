"""
Facial Verification Pipeline - enrol a face, then check whether a later face is the same person.

    python3 verify.py download-models
    python3 verify.py enrol alice --source 0 --show          (webcam)
    python3 verify.py verify alice --source 0 --show
    python3 verify.py landmarks --source samples/person_a.jpg --output out.jpg
    python3 verify.py list
    python3 verify.py delete alice

--source can be a camera number (0), a video file, a folder of images or one image.
No image or raw embedding is ever saved: only a keyed hash (see facepipe/template.py).
"""

import argparse
import os
import sys

import cv2
import numpy as np

from facepipe import models, pipeline, template
from facepipe.detector import LANDMARK_NAMES, FaceAnalyser, draw
from facepipe.quality import clean_frame

HERE = os.path.dirname(os.path.abspath(__file__))


def describe(stats):
    return ("{frames} frame(s) read: {used} used, {no_face} without a face, {blurred} too blurry; "
            "{denoised} denoised, {contrast_boosted} brightened".format(**stats))


def gather(args, wanted):
    analyser = FaceAnalyser()
    embeddings, stats = pipeline.collect_embeddings(
        analyser, pipeline.frames_from(args.source), wanted=wanted, show=args.show)
    print("[*] " + describe(stats))
    if not embeddings:
        sys.exit("[!] No usable face was found. Face the camera in good light and try again.")
    return embeddings


def open_store(args):
    return template.TemplateStore(args.store, template.load_or_create_key(args.key_file))


def cmd_download(args):
    models.download_all()
    print("[✓] Models are ready in " + models.MODEL_DIR)


def cmd_enrol(args):
    template.check_name(args.name)
    store = open_store(args)
    embeddings = gather(args, args.frames)
    store.enrol(args.name, pipeline.average(embeddings), frames=len(embeddings))
    print("[✓] Enrolled '{}' from {} frame(s). Only a {}-bit keyed hash was stored.".format(
        args.name, len(embeddings), template.N_BITS))


def cmd_verify(args):
    store = open_store(args)
    store.load(args.name)                                   # fail early if the name is unknown
    embeddings = gather(args, args.frames)
    distances = [store.compare(args.name, embedding) for embedding in embeddings]
    score = float(np.median(distances))
    matched = score <= args.threshold
    print("[*] Hash distance {:.3f} (threshold {:.3f}; same face is near 0.2, a stranger near 0.5)".format(
        score, args.threshold))
    print("[✓] MATCH: this is '{}'.".format(args.name) if matched
          else "[✗] NO MATCH: this is not '{}'.".format(args.name))
    sys.exit(0 if matched else 1)


def cmd_landmarks(args):
    analyser = FaceAnalyser()
    frame = next(pipeline.frames_from(args.source))
    cleaned, report = clean_frame(frame)
    faces = analyser.detect(cleaned)
    if not faces:
        sys.exit("[!] No face found.")
    print("[*] Frame noise {:.1f}, brightness {:.0f}, fixes applied: {}".format(
        report["noise"], report["brightness"], ", ".join(report["fixes"]) or "none"))
    canvas = cleaned
    for number, face in enumerate(faces, start=1):
        print("Face {}: box {} confidence {:.2f}".format(number, face.box, face.confidence))
        for name, (x, y) in zip(LANDMARK_NAMES, face.landmarks):
            print("    {:<20} ({:.0f}, {:.0f})".format(name, x, y))
        canvas = draw(canvas, face, "face {} ({:.2f})".format(number, face.confidence))
    cv2.imwrite(args.output, canvas)
    print("[✓] Annotated image saved to " + os.path.abspath(args.output))


def cmd_list(args):
    names = open_store(args).names()
    print("\n".join(names) if names else "Nobody is enrolled yet.")


def cmd_delete(args):
    open_store(args).delete(args.name)
    print("[✓] Deleted the template for '{}'.".format(args.name))


def main():
    parser = argparse.ArgumentParser(description="Face verification with protected biometric templates.")
    parser.add_argument("--store", default=os.path.join(HERE, "templates"), help="folder for the stored hashes")
    parser.add_argument("--key-file", default=os.path.join(HERE, "facepipe.key"), help="file holding the secret key")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_source(sub, frames):
        sub.add_argument("--source", default="0", help="camera number, video file, image folder or image")
        sub.add_argument("--frames", type=int, default=frames, help="good frames to collect")
        sub.add_argument("--show", action="store_true", help="show the live feed with landmarks")

    commands.add_parser("download-models", help="fetch and verify the model files").set_defaults(run=cmd_download)

    enrol = commands.add_parser("enrol", help="store a protected template for a person")
    enrol.add_argument("name")
    add_source(enrol, 10)
    enrol.set_defaults(run=cmd_enrol)

    verify = commands.add_parser("verify", help="check a face against an enrolled person")
    verify.add_argument("name")
    add_source(verify, 5)
    verify.add_argument("--threshold", type=float, default=template.MATCH_THRESHOLD)
    verify.set_defaults(run=cmd_verify)

    marks = commands.add_parser("landmarks", help="draw the face box and landmarks on one frame")
    marks.add_argument("--source", default="0")
    marks.add_argument("--output", default="landmarks.jpg")
    marks.set_defaults(run=cmd_landmarks)

    commands.add_parser("list", help="show who is enrolled").set_defaults(run=cmd_list)
    remove = commands.add_parser("delete", help="delete a person's template")
    remove.add_argument("name")
    remove.set_defaults(run=cmd_delete)

    args = parser.parse_args()
    try:
        args.run(args)
    except (template.TemplateError, pipeline.SourceError, FileNotFoundError, RuntimeError) as error:
        sys.exit("[!] {}".format(error))


if __name__ == "__main__":
    main()
