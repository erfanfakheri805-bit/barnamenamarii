# -*- coding: utf-8 -*-
"""
Desktop tests (no Kivy / Android needed):   python tests/test_engine.py

* classify_hand / GestureTracker on synthetic landmark sets
* SharedBuffer (mmap) round trip
* persian.fa() sanity
* end-to-end: a synthetic "person" walks in front of a textured background;
  after the background is learned a fist makes the person vanish (checked against
  the true background), an open hand brings them back, pixels outside the matte
  stay untouched.  Timing at 720p / 1080p is printed for reference only.
"""
import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import ar_engine as ae                                     # noqa: E402
from persian import fa, to_fa_digits                       # noqa: E402

RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ hands --
def synth_hand(kind, noise=0.008):
    pts = np.zeros((21, 3), np.float32)
    angles = {"index": -15, "middle": -3, "ring": 9, "pinky": 21}
    lengths = {"index": 0.95, "middle": 1.0, "ring": 0.95, "pinky": 0.85}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for name, ang in angles.items():
        th = np.deg2rad(ang)
        u = np.array([np.sin(th), np.cos(th)])
        L = lengths[name]
        extended = kind == "open" or (kind == "point" and name == "index") or (kind == "peace" and name in ("index", "middle"))
        if extended:
            radii, zs = (0.5, 0.7 * L, 0.82 * L, 0.92 * L), (0, 0, 0, 0)
        else:
            radii, zs = (0.5, 0.55, 0.5, 0.42), (0, -0.2, -0.25, -0.15)
        for k in range(4):
            x, y = u * radii[k]
            pts[base[name] + k] = (x, y, zs[k])
    for i, (x, y) in enumerate(((-0.1, 0.15), (-0.3, 0.3), (-0.45, 0.4), (-0.55, 0.5))):
        pts[1 + i] = (x, y, 0)
    pts += RNG.normal(0, noise, pts.shape).astype(np.float32)
    out = np.empty_like(pts)
    out[:, 0] = 0.5 + pts[:, 0] * 0.3
    out[:, 1] = 0.8 - pts[:, 1] * 0.3
    out[:, 2] = pts[:, 2] * 0.3
    return out


def test_classify():
    for kind, want in (("open", "open"), ("fist", "fist"), ("point", "other"), ("peace", "other")):
        got = [ae.classify_hand(synth_hand(kind), 270, 480) for _ in range(300)]
        ok = got.count(want) / len(got)
        print("  %-6s -> %-5s %.1f%%" % (kind, want, 100 * ok))
        assert ok > 0.97, (kind, ok)


def test_tracker():
    cfg = ae.EngineConfig()
    tr = ae.GestureTracker(cfg)
    t = 0.0
    assert not tr.update("fist", t)
    assert tr.update("fist", t + .03)                       # 2 confirmations
    assert tr.update("other", t + .06)                      # 'other' keeps state
    assert tr.update(None, t + .5)                          # hand lost, still inside hold
    assert not tr.update(None, t + 0.06 + cfg.hand_hold_s + .1)
    tr.update("open", 5.0)
    assert not tr.update("open", 5.03)
    print("  tracker ok")


def test_shared_buffer():
    path = os.path.join(tempfile.mkdtemp(), "x.rgba")
    sb = ae.SharedBuffer(path, 1000)
    sb.u8[:4] = (1, 2, 3, 4)
    other = np.frombuffer(sb.mm, dtype=np.uint8, count=4)
    assert tuple(other) == (1, 2, 3, 4)
    del other
    sb.close()
    print("  shared buffer ok")


def test_persian():
    s = fa("نامرئی‌ساز")
    assert "\u200c" not in s and all(ord(c) > 0xFB00 for c in s), s
    assert to_fa_digits("30 FPS") == "۳۰ FPS"
    assert fa("FPS 30") == "FPS ۳۰"
    print("  persian ok:", s)


# --------------------------------------------------------------- end to end --
def make_background(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([60 + 120 * xx / w, 80 + 100 * yy / h, 140 + 60 * np.sin(xx / 90.0)], axis=2)
    bg += RNG.normal(0, 4, bg.shape)
    img = np.clip(bg, 0, 255).astype(np.uint8)
    for i in range(14):                                     # texture / landmarks
        c = tuple(int(v) for v in RNG.integers(30, 230, 3))
        p = (int(RNG.integers(0, w)), int(RNG.integers(0, h)))
        if i % 2:
            cv2.circle(img, p, int(RNG.integers(20, 90)), c, -1)
        else:
            cv2.rectangle(img, p, (p[0] + int(RNG.integers(40, 200)), p[1] + int(RNG.integers(40, 200))), c, -1)
    return img


def draw_person(frame, cx, cy, scale):
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.ellipse(mask, (cx, cy - int(250 * scale)), (int(70 * scale), int(90 * scale)), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (cx - int(130 * scale), cy - int(160 * scale)), (cx + int(130 * scale), cy + int(420 * scale)), 255, -1)
    body = np.zeros_like(frame)
    body[:] = (200, 40, 60)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    body[..., 1] = (60 + 90 * ((xx // 24 + yy // 24) % 2)).astype(np.uint8)       # checker shirt
    frame[mask > 0] = body[mask > 0]
    return mask


class FakeBackend(ae.VisionBackend):
    """Stands in for MediaPipe: the harness provides the ground-truth person mask
    (as small float image) and the hand pose; segmentation lags like a real model."""
    name = "fake"
    has_hands = True
    has_seg = True

    def __init__(self, *_):
        self.gt_small = None
        self.hand = None

    def detect_hand(self, rgb, ts_ms):
        return None if self.hand is None else synth_hand(self.hand)

    def segment(self, rgb, ts_ms):
        g = self.gt_small
        if g is None:
            return np.zeros(rgb.shape[:2], np.float32)
        return cv2.GaussianBlur(g, (0, 0), 1.6)

    def feed(self, gt_u8, size):
        self.gt_small = cv2.resize(gt_u8, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def run_e2e(h, w, label):
    bg_true = make_background(h, w)
    scale = h / 1920.0
    sh, sw = ae.InvisibilityEngine(ae.NullBackend())._ml_shape(h, w)
    backend = FakeBackend()
    eng = ae.InvisibilityEngine(backend)
    eng.start()
    times = []
    results = {}

    def frame_at(t):
        f = bg_true.copy()
        cx = int(w / 2 + 0.33 * w * np.sin(2 * np.pi * t / 4.0))
        m = draw_person(f, cx, int(h * 0.5), scale)
        return f, m

    t0 = time.monotonic()
    phase_end = {"learn": 4.2, "fist": 6.2, "open": 8.2}
    last_info = None
    while True:
        t = time.monotonic() - t0
        if t > phase_end["open"]:
            break
        backend.hand = None if t < phase_end["learn"] else ("fist" if t < phase_end["fist"] else "open")
        f, gt = frame_at(t)
        live = f.copy()
        backend.feed(gt, (sw, sh))
        s = time.perf_counter()
        out, info = eng.process(f)
        times.append((time.perf_counter() - s) * 1000)
        last_info = info
        if 5.7 < t < 6.2 and "fist" not in results and info["progress"] > 0.999:
            core = cv2.erode(gt, np.ones((15, 15), np.uint8)) > 0
            err = np.abs(out.astype(np.int16) - bg_true.astype(np.int16)).mean(axis=2)[core].mean()
            far = cv2.dilate(gt, np.ones((int(120 * scale) | 1,) * 2, np.uint8)) == 0
            untouched = np.abs(out.astype(np.int16) - live.astype(np.int16)).max(axis=2)[far].max()
            results["fist"] = (err, untouched, core.sum())
        if t > 7.9 and "open" not in results and info["progress"] < 0.001:
            results["open"] = float(np.abs(out.astype(np.int16) - live.astype(np.int16)).max())
        time.sleep(max(0.0, 1 / 30.0 - (time.perf_counter() - s)))
    eng.stop()
    ms = np.array(times)
    print("  [%s] process(): mean %.1f ms  p95 %.1f ms  (this PC, not a phone)" % (label, ms.mean(), np.percentile(ms, 95)))
    print("  [%s] coverage %.0f%%  invisible-phase error vs true background: %.2f levels" %
          (label, 100 * last_info["coverage"], results["fist"][0]))
    assert "fist" in results, "never reached full invisibility"
    assert results["fist"][0] < 8.0, results["fist"]
    assert results["fist"][1] <= 2, "pixels far from the person were modified: %s" % (results["fist"],)
    assert "open" in results and results["open"] == 0.0, results.get("open")


# ---------------------------------------------------------------- scenarios --
def _scenario(h, w, secs, hand_fn, person_fn, backend=None, manual=None, bg_fn=None):
    bg_true = make_background(h, w)
    backend = backend or FakeBackend()
    eng = ae.InvisibilityEngine(backend)
    sh, sw = eng._ml_shape(h, w)
    eng.start()
    log = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        t = time.monotonic() - t0
        if hasattr(backend, "hand"):
            backend.hand = hand_fn(t)
        if manual is not None:
            eng.set_manual(manual(t))
        f = (bg_fn(bg_true, t) if bg_fn else bg_true).copy()
        gt = person_fn(f, t, h, w)
        if hasattr(backend, "feed"):
            backend.feed(gt, (sw, sh))
        s = time.perf_counter()
        out, info = eng.process(f)
        log.append((t, info, gt, out, bg_true))
        time.sleep(max(0.0, 1 / 30.0 - (time.perf_counter() - s)))
    eng.stop()
    return log, eng


def _core_error(entry):
    _, _, gt, out, bg_true = entry
    core = cv2.erode(gt, np.ones((15, 15), np.uint8)) > 0
    return float(np.abs(out.astype(int) - bg_true.astype(int)).mean(axis=2)[core].mean())


def test_scenarios():
    scale = 1280 / 1920.0
    print("  person never leaves the frame -> unseen pixels are inpainted")
    log, eng = _scenario(1280, 720, 5.0, lambda t: None if t < 3 else "fist",
                         lambda f, t, h, w: draw_person(f, w // 2, h // 2, scale))
    assert log[-1][1]["progress"] > 0.99 and eng.bg.fill_small is not None
    print("    coverage %.0f%%, error inside person %.1f levels (coarse fill, expected)" %
          (100 * log[-1][1]["coverage"], _core_error(log[-1])))

    print("  camera moves while visible -> background is reset")
    def walker(f, t, h, w):
        return draw_person(f, int(w / 2 + 0.3 * w * np.sin(2 * np.pi * t / 4)), h // 2, scale)

    def shift_bg(bg, t):
        if t < 3.0:
            return bg
        return cv2.warpAffine(bg, np.float32([[1, 0, 70], [0, 1, 40]]), (bg.shape[1], bg.shape[0]),
                              borderMode=cv2.BORDER_REFLECT)
    log, _ = _scenario(1280, 720, 6.0, lambda t: None, walker, bg_fn=shift_bg)
    resets = [x[0] for x in log if x[1]["reset"]]
    assert resets and 3.0 < resets[0] < 5.6, resets
    print("    reset detected %.1fs after the move" % (resets[0] - 3.0))

    print("  no MediaPipe (classic mode): calibrate, then manual toggle")
    def enters(f, t, h, w):
        return np.zeros(f.shape[:2], np.uint8) if t < 3.6 else draw_person(f, w // 2, h // 2, scale)
    log, _ = _scenario(1280, 720, 7.0, lambda t: None, enters, backend=ae.NullBackend("test"),
                       manual=lambda t: t > 5.0)
    assert log[-1][1]["progress"] > 0.99 and _core_error(log[-1]) < 6
    print("    ok (error %.2f)" % _core_error(log[-1]))

    print("  models that keep failing are switched off at runtime")

    class Broken(FakeBackend):
        def segment(self, rgb, ts_ms):
            raise RuntimeError("model file is broken")

        def detect_hand(self, rgb, ts_ms):
            raise RuntimeError("hand model is broken")
    log, eng = _scenario(1280, 720, 3.5, lambda t: None,
                         lambda f, t, h, w: np.zeros(f.shape[:2], np.uint8), backend=Broken())
    last = log[-1][1]
    assert not last["has_seg"] and not last["has_hands"], last
    assert "failed at runtime" in eng.backend.reason
    print("    ok -> engine fell back to classic mode (%s)" % eng.backend.reason)

    print("  camera resolution switch")
    eng = ae.InvisibilityEngine(FakeBackend())
    eng.start()
    base = make_background(1920, 1080)
    for hh, ww in ((1280, 720), (1920, 1080), (1280, 720)):
        for _ in range(8):
            eng.process(cv2.resize(base, (ww, hh)))
            time.sleep(0.02)
    eng.stop()
    print("    ok")


def test_cv2_features():
    a = np.full((8, 8, 3), 100, np.uint8)
    b = np.full((8, 8, 3), 200, np.uint8)
    w = np.full((8, 8), 0.25, np.float32)
    out = cv2.blendLinear(a, b, 1 - w, w)
    assert abs(int(out[0, 0, 0]) - 125) <= 1
    m = cv2.multiply(a, (1.5, 1.0, 0.5, 0.0), dtype=cv2.CV_8U)
    assert tuple(m[0, 0]) == (150, 100, 50), tuple(m[0, 0])
    print("  opencv blendLinear / per-channel multiply ok")


def test_benchmark():
    ms = ae.benchmark_pipeline(1280, 720)
    assert ms is not None and 1.0 < ms < 500.0, ms
    print("  device benchmark 720p: %.1f ms per frame" % ms)


if __name__ == "__main__":
    print("opencv", cv2.__version__)
    for name, fn in (("cv2 features", test_cv2_features), ("hand classification", test_classify),
                     ("gesture tracker", test_tracker), ("shared buffer", test_shared_buffer),
                     ("benchmark", test_benchmark),
                     ("persian", test_persian)):
        print(name)
        fn()
    print("end-to-end 720p")
    run_e2e(1280, 720, "720p")
    print("end-to-end 1080p")
    run_e2e(1920, 1080, "1080p")
    print("scenarios")
    test_scenarios()
    print("ALL TESTS PASSED")
