#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/fetch_assets.py - downloads the binary assets that are not stored in git.

  assets/fonts/Vazirmatn-Regular.ttf , Vazirmatn-Bold.ttf   (SIL OFL 1.1)
  assets/models/hand_landmarker.task                         (MediaPipe, Apache-2.0)
  assets/models/selfie_segmenter.tflite                      (MediaPipe, Apache-2.0)

Usage:   python scripts/fetch_assets.py [--force] [--skip-fonts] [--skip-models] [--verify-font]

--verify-font  checks (needs `pip install fonttools`) that the font really contains the
               Arabic Presentation Forms the Kivy text renderer needs.  If it does not, the
               font is removed and the app silently uses the bundled DejaVu Sans instead,
               so Persian text is never rendered broken.
Exit code 1 when a required model could not be downloaded.
"""
import argparse
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_TAG = "v33.003"
FONT_FILES = ("Vazirmatn-Regular.ttf", "Vazirmatn-Bold.ttf")
FONT_URLS = (
    "https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@" + FONT_TAG + "/fonts/ttf/{name}",
    "https://raw.githubusercontent.com/rastikerdar/vazirmatn/" + FONT_TAG + "/fonts/ttf/{name}",
    "https://github.com/rastikerdar/vazirmatn/raw/" + FONT_TAG + "/fonts/ttf/{name}",
)
MODELS = {
    "hand_landmarker.task": (
        ["https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
         "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"],
        5_000_000, b"PK"),
    "selfie_segmenter.tflite": (
        ["https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
         "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"],
        100_000, None),
}
# glyphs the shaper in persian.py emits for typical UI text
REQUIRED_CODEPOINTS = [0xFEB3, 0xFEFB, 0xFEFC, 0xFB56, 0xFBFC, 0xFB8E, 0xFB92, 0xFB7A, 0xFB8A,
                       0xFEE1, 0xFE8D, 0xFEE9, 0xFEE0, 0x06F0, 0x06F9]


def download(urls, dest, min_size, magic=None, tries=3):
    for url in urls:
        for attempt in range(1, tries + 1):
            try:
                print("  GET %s" % url)
                req = urllib.request.Request(url, headers={"User-Agent": "InvisibilityAR-build/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                if len(data) < min_size:
                    raise IOError("file too small (%d bytes)" % len(data))
                if magic and not data.startswith(magic):
                    raise IOError("unexpected file signature")
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
            if not download([u.format(name=name) for u in FONT_URLS], dest, 50_000, None):
                print("  !! could not download %s - the app will use the bundled DejaVu Sans" % name)
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
        for name, (urls, min_size, magic) in MODELS.items():
            dest = os.path.join(ROOT, "assets", "models", name)
            if os.path.exists(dest) and not args.force:
                print("  have %s" % name)
                continue
            if not download(urls, dest, min_size, magic):
                failed.append(name)

    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
