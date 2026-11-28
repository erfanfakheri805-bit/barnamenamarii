#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/fetch_assets.py - downloads the binary assets that are not stored in git.

  assets/fonts/Vazirmatn-Regular.ttf , Vazirmatn-Bold.ttf   (SIL OFL 1.1)
  assets/models/hand_landmarker.task                         (MediaPipe, Apache-2.0)
  assets/models/selfie_segmenter.tflite                      (MediaPipe, Apache-2.0)

Usage:   python scripts/fetch_assets.py [--force] [--skip-fonts] [--skip-models] [--verify-font] [--strict]

--verify-font  checks (needs `pip install fonttools`) that the font really contains the
               Arabic Presentation Forms the Kivy text renderer needs.  If it does not, the
               font is removed and the app silently uses the bundled DejaVu Sans instead.
--strict       exit code 1 when a model could not be downloaded.  Without it the script only
               warns: the app still builds and falls back to its "compatible mode" (manual toggle)
               when a model is missing.

Validation is deliberately tolerant: it checks the size range and prints the first bytes /
HTTP headers of every download, but never rejects a file only because of its signature
(a previous version did that for hand_landmarker.task and broke the build).
"""
import argparse
import gzip
import io
import os
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_TAG = "v33.003"
FONT_FILES = ("Vazirmatn-Regular.ttf", "Vazirmatn-Bold.ttf")
FONT_URLS = (
    "https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@" + FONT_TAG + "/fonts/ttf/{name}",
    "https://raw.githubusercontent.com/rastikerdar/vazirmatn/" + FONT_TAG + "/fonts/ttf/{name}",
    "https://github.com/rastikerdar/vazirmatn/raw/" + FONT_TAG + "/fonts/ttf/{name}",
)
# name -> (urls, min_size, max_size)
MODELS = {
    "hand_landmarker.task": (
        ["https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
         "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"],
        3_000_000, 60_000_000),
    "selfie_segmenter.tflite": (
        ["https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
         "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"],
        100_000, 20_000_000),
}
# glyphs the shaper in persian.py emits for typical UI text
REQUIRED_CODEPOINTS = [0xFEB3, 0xFEFB, 0xFEFC, 0xFB56, 0xFBFC, 0xFB8E, 0xFB92, 0xFB7A, 0xFB8A,
                       0xFEE1, 0xFE8D, 0xFEE9, 0xFEE0, 0x06F0, 0x06F9]


def describe(data):
    head = data[:8]
    return "%d bytes, first bytes %s (%r)" % (len(data), head.hex(" "), head)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "InvisibilityAR-build/1.1", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = resp.read()
        info = "HTTP %s, content-type=%s, content-encoding=%s" % (
            resp.status, resp.headers.get("Content-Type"), resp.headers.get("Content-Encoding"))
    if data[:2] == b"\x1f\x8b":                       # a gzip body that was not decoded for us
        try:
            data = gzip.decompress(data)
            info += " (gunzipped)"
        except Exception:
            pass
    return data, info


def download(urls, dest, min_size, max_size=None, tries=3):
    for url in urls:
        for attempt in range(1, tries + 1):
            try:
                print("  GET %s" % url)
                data, info = fetch(url)
                print("      %s | %s" % (info, describe(data)))
                if len(data) < min_size:
                    raise IOError("file too small (%d bytes, expected >= %d)" % (len(data), min_size))
                if max_size and len(data) > max_size:
                    raise IOError("file too large (%d bytes)" % len(data))
                if data[:5].lower() in (b"<!doc", b"<html", b"<?xml"):
                    raise IOError("got a web page / XML error instead of the file")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tmp = dest + ".part"
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, dest)
                print("  OK  %s (%.1f KB)" % (os.path.relpath(dest, ROOT), len(data) / 1024.0))
                return True
            except Exception as exc:
                print("  ..  attempt %d failed: %s" % (attempt, exc))
                time.sleep(1.5 * attempt)
    return False


def sanity_note(path):
    """Informational only - never rejects the file."""
    name = os.path.basename(path)
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if name.endswith(".task"):
            kind = "zip bundle" if zipfile.is_zipfile(io.BytesIO(data)) else (
                "tflite inside" if b"TFL3" in data[:4_000_000] else "unknown container")
            print("  note: %s looks like a %s" % (name, kind))
        elif name.endswith(".tflite"):
            print("  note: %s has TFLite id %r" % (name, data[4:8]))
    except Exception as exc:
        print("  note: could not inspect %s (%s)" % (name, exc))


def font_ok(path):
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        print("  (fonttools missing - coverage check skipped; pip install fonttools)")
        return True
    cmap = TTFont(path).getBestCmap()
    missing = [hex(cp) for cp in REQUIRED_CODEPOINTS if cp not in cmap]
    if missing:
        print("  !! %s lacks %s" % (os.path.basename(path), ", ".join(missing)))
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-fonts", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--verify-font", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    failed = []

    if not args.skip_fonts:
        print("Fonts")
        fonts_dir = os.path.join(ROOT, "assets", "fonts")
        for name in FONT_FILES:
            dest = os.path.join(fonts_dir, name)
            if os.path.exists(dest) and not args.force:
                print("  have %s" % name)
                continue
            if not download([u.format(name=name) for u in FONT_URLS], dest, 50_000, 5_000_000):
                print("::warning::could not download %s - the app will use the bundled DejaVu Sans" % name)
        if args.verify_font:
            reg = os.path.join(fonts_dir, FONT_FILES[0])
            if os.path.exists(reg) and not font_ok(reg):
                print("  Vazirmatn does not cover the needed presentation forms -> removing, DejaVu Sans fallback is used")
                for name in FONT_FILES:
                    p = os.path.join(fonts_dir, name)
                    if os.path.exists(p):
                        os.remove(p)

    if not args.skip_models:
        print("MediaPipe models")
        for name, (urls, min_size, max_size) in MODELS.items():
            dest = os.path.join(ROOT, "assets", "models", name)
            if os.path.exists(dest) and not args.force:
                print("  have %s" % name)
                continue
            if download(urls, dest, min_size, max_size):
                sanity_note(dest)
            else:
                failed.append(name)

    if failed:
        msg = ("could not download: " + ", ".join(failed) +
               " - the app will start in compatible mode (manual invisible button) until the model is added")
        print("::warning::" + msg)
        print("WARNING: " + msg)
        return 1 if args.strict else 0
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
