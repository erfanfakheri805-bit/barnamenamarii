# -*- coding: utf-8 -*-
"""
ar_engine.py - Invisibility AR vision engine (OpenCV + MediaPipe).

Pipeline (one call to InvisibilityEngine.process() per camera frame):

    full-res RGB frame ──► downscale (<=480 px) ──► MediaPipe workers (async, latest-frame-wins)
            │                                         ├─ HandLandmarker  -> fist / open
            │                                         └─ ImageSegmenter  -> person probability
            ▼
    person matte  (EMA -> contrast -> guided filter -> expand -> Gaussian feather)
            ▼
    background model (masked running average, only learns pixels that are NOT person)
            ▼
    composite on the ORIGINAL full-res frame (only inside the person's bounding box)

MediaPipe on Android
--------------------
The PyPI `mediapipe` wheel does not exist for Android, so on a phone the official
MediaPipe *Tasks* AAR (com.google.mediapipe:tasks-vision, added through
buildozer's `android.gradle_dependencies`) is driven through pyjnius.
pyjnius converts Java arrays by value (slow for megabytes), therefore pixel data
and masks never cross the bridge as arrays: they travel through *file-backed
shared memory* - the same file is mmap()-ed by Python and by a Java
MappedByteBuffer (see SharedBuffer).  Only small values (21 landmarks) are read
through normal pyjnius calls.

On a desktop the regular `mediapipe` Python package is used (development aid).
If no MediaPipe backend can be created the engine degrades gracefully to a
classical background-subtraction mask and a manual invisible toggle.
"""
from __future__ import annotations

import logging
import mmap
import os
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("invisibility.engine")

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


class EngineConfig:
    ml_long_side = 480          # MediaPipe input long side (px)
    matte_lo = 0.30             # person probability mapped to alpha 0
    matte_hi = 0.70             # person probability mapped to alpha 1
    mask_new_weight = 0.6       # temporal smoothing weight of a new mask
    guided_radius = 4           # guided-filter radius (work resolution px)
    guided_eps = 2e-3
    expand_frac = 0.006         # matte dilation (fraction of work width)
    feather_frac = 0.004        # gaussian feather sigma (fraction of work width)
    transition_s = 0.45         # fade duration visible <-> invisible
    hand_hold_s = 1.2           # keep state this long after the hand vanishes
    fist_confirm = 2            # consecutive detections needed -> invisible
    open_confirm = 2            # consecutive detections needed -> visible
    bg_learn = 0.04             # running-average rate for known background pixels
    bg_every = 3                # update the background every N frames
    bg_known_step = 0.34        # confidence gained per observation (>0.5 == known)
    move_thresh = 10.0          # median |frame - background| (0..255) that means "camera moved"
    move_steps = 6              # consecutive checks above the threshold -> reset
    calib_seconds = 3.0         # classic (no-ML) mode: initial background capture
    aug_thresh = 24             # |frame - background| that counts as person (hybrid matte)
    move_gate = 14              # frame-to-frame change that blocks background learning
    shimmer = True              # accent rim light while fading
    accent = (56, 189, 248)     # #38BDF8


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


class FpsMeter:
    def __init__(self, tau=0.5):
        self.tau = tau
        self.value = 0.0
        self._last = None

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                a = min(1.0, dt / self.tau)
                self.value = inst if self.value == 0.0 else self.value + a * (inst - self.value)
        self._last = now
        return self.value

    def reset(self):
        self.value = 0.0
        self._last = None


class LatestSlot:
    """Thread-safe one-item mailbox: put() overwrites, get() blocks (latest wins)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None
        self._closed = False
        self.dropped = 0

    def put(self, item):
        with self._cv:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._cv.notify()

    def get(self, timeout=0.25):
        with self._cv:
            if self._item is None and not self._closed:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item

    def get_nowait(self):
        with self._cv:
            item, self._item = self._item, None
            return item

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self):
        return self._closed


class _Worker(threading.Thread):
    """Runs fn(item) on the newest submitted item; keeps only the latest result."""

    def __init__(self, name, fn):
        super().__init__(name=name, daemon=True)
        self._fn = fn
        self._slot = LatestSlot()
        self._lock = threading.Lock()
        self._res = None
        self._seq = 0
        self._errors = 0
        self.ok = 0                      # successful jobs so far
        self.fps = FpsMeter(1.0)

    def submit(self, item):
        self._slot.put(item)

    def latest(self):
        with self._lock:
            return self._res, self._seq

    def stop(self):
        self._slot.close()

    def run(self):
        while not self._slot.closed:
            item = self._slot.get(0.25)
            if item is None:
                continue
            try:
                res = self._fn(item)
                self._errors = 0
                if res is not None:
                    self.ok += 1
            except Exception:                               # never kill the thread
                self._errors += 1
                if self._errors in (1, 10, 100):
                    log.exception("%s failed (%d)", self.name, self._errors)
                res = None
                time.sleep(min(0.5, 0.02 * self._errors))
            with self._lock:
                self._res = res
                self._seq += 1
            self.fps.tick()


class SharedBuffer:
    """File-backed memory visible to Python (mmap) and optionally Java (MappedByteBuffer).

    Both sides map the same file with MAP_SHARED, so bulk pixel / mask transfer needs
    no pyjnius array conversion at all.
    """

    def __init__(self, path, nbytes, java=False):
        self.path = path
        self.nbytes = int(nbytes)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(self.nbytes)
        self._f = open(path, "r+b")
        self.mm = mmap.mmap(self._f.fileno(), self.nbytes)
        self.u8 = np.frombuffer(self.mm, dtype=np.uint8)
        self.jbuf = None
        self._raf = None
        if java:
            from jnius import autoclass                      # noqa: WPS433 (Android only)
            RandomAccessFile = autoclass("java.io.RandomAccessFile")
            MapMode = autoclass("java.nio.channels.FileChannel$MapMode")
            ByteOrder = autoclass("java.nio.ByteOrder")
            self._raf = RandomAccessFile(path, "rw")
            channel = self._raf.getChannel()
            self.jbuf = channel.map(MapMode.READ_WRITE, 0, self.nbytes)
            self.jbuf.order(ByteOrder.nativeOrder())

    def close(self):
        self.jbuf = None
        try:
            if self._raf is not None:
                self._raf.close()
        except Exception:
            pass
        self.u8 = None
        try:
            self.mm.close()
        except Exception:
            pass
        try:
            self._f.close()
            os.remove(self.path)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# hand gesture
# --------------------------------------------------------------------------- #

_FINGERS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))   # (MCP, PIP, TIP)


def classify_hand(lm, width=1.0, height=1.0):
    """21 MediaPipe landmarks (x, y, z normalised) -> 'fist' | 'open' | 'other'.

    Rotation and scale invariant: compares each fingertip's distance to the wrist
    with its MCP / PIP joint distances (3-D, aspect corrected).
    """
    p = np.asarray(lm, dtype=np.float32).reshape(21, 3) * np.array([width, height, width], np.float32)
    wrist = p[0]
    extended = curled = 0
    for mcp, pip, tip in _FINGERS:
        d_mcp = float(np.linalg.norm(p[mcp] - wrist)) + 1e-6
        d_pip = float(np.linalg.norm(p[pip] - wrist))
        d_tip = float(np.linalg.norm(p[tip] - wrist))
        ratio = d_tip / d_mcp
        if ratio > 1.5 and d_tip > d_pip * 1.08:
            extended += 1
        elif ratio < 1.28 or d_tip < d_pip * 0.97:
            curled += 1
    if curled >= 3 and extended == 0:
        return "fist"
    if extended >= 3 and curled == 0:
        return "open"
    return "other"


class GestureTracker:
    """Debounces gesture detections into the invisible / visible target state."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.invisible = False
        self.hand = "none"
        self._fist = 0
        self._open = 0
        self._last_seen = 0.0

    def reset(self):
        self.invisible = False
        self.hand = "none"
        self._fist = self._open = 0

    def update(self, label, now):
        """Call once per NEW detection result (label: fist/open/other/None)."""
        if label is None:
            self._fist = self._open = 0
            if self.invisible and now - self._last_seen > self.cfg.hand_hold_s:
                self.invisible = False
            if not self.invisible:
                self.hand = "none"
            return self.invisible
        self._last_seen = now
        if label == "fist":
            self._fist += 1
            self._open = 0
            if self._fist >= self.cfg.fist_confirm:
                self.invisible = True
                self.hand = "fist"
        elif label == "open":
            self._open += 1
            self._fist = 0
            if self._open >= self.cfg.open_confirm:
                self.invisible = False
                self.hand = "open"
        return self.invisible


# --------------------------------------------------------------------------- #
# vision backends
# --------------------------------------------------------------------------- #


class VisionBackend:
    name = "none"
    has_hands = False
    has_seg = False
    reason = ""

    def open(self):
        pass

    def detect_hand(self, rgb, ts_ms):          # -> ndarray (21, 3) | None
        return None

    def segment(self, rgb, ts_ms):              # -> float32 (H, W) person probability | None
        return None

    def close(self):
        pass


class NullBackend(VisionBackend):
    name = "classic"

    def __init__(self, reason=""):
        self.reason = reason


class PythonMediaPipeBackend(VisionBackend):
    """Desktop development backend (pip install mediapipe)."""

    name = "mediapipe-python"

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self._hands = self._seg = None
        self._hands_kind = self._seg_kind = None
        self._mp = None

    def open(self):
        import mediapipe as mp                                # noqa: WPS433
        self._mp = mp
        errors = []
        hand_path = os.path.join(self.models_dir, "hand_landmarker.task")
        seg_path = os.path.join(self.models_dir, "selfie_segmenter.tflite")
        try:
            if os.path.isfile(hand_path):
                from mediapipe.tasks import python as mpp
                from mediapipe.tasks.python import vision as mpv
                opts = mpv.HandLandmarkerOptions(
                    base_options=mpp.BaseOptions(model_asset_path=hand_path),
                    running_mode=mpv.RunningMode.VIDEO, num_hands=1,
                    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5)
                self._hands = mpv.HandLandmarker.create_from_options(opts)
                self._hands_kind = "tasks"
            else:
                self._hands = mp.solutions.hands.Hands(
                    max_num_hands=1, model_complexity=0,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self._hands_kind = "solutions"
            self.has_hands = True
        except Exception as exc:
            errors.append("hands: %s" % exc)
        try:
            if os.path.isfile(seg_path):
                from mediapipe.tasks import python as mpp
                from mediapipe.tasks.python import vision as mpv
                opts = mpv.ImageSegmenterOptions(
                    base_options=mpp.BaseOptions(model_asset_path=seg_path),
                    running_mode=mpv.RunningMode.VIDEO,
                    output_confidence_masks=True, output_category_mask=False)
                self._seg = mpv.ImageSegmenter.create_from_options(opts)
                self._seg_kind = "tasks"
            else:
                self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
                self._seg_kind = "solutions"
            self.has_seg = True
        except Exception as exc:
            errors.append("segmentation: %s" % exc)
        self.reason = "; ".join(errors)

    def detect_hand(self, rgb, ts_ms):
        rgb = np.ascontiguousarray(rgb)
        if self._hands_kind == "tasks":
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self._hands.detect_for_video(img, int(ts_ms))
            if not res.hand_landmarks:
                return None
            pts = res.hand_landmarks[0]
        else:
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            pts = res.multi_hand_landmarks[0].landmark
        return np.array([[q.x, q.y, q.z] for q in pts], dtype=np.float32)

    def segment(self, rgb, ts_ms):
        rgb = np.ascontiguousarray(rgb)
        if self._seg_kind == "tasks":
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res = self._seg.segment_for_video(img, int(ts_ms))
            masks = res.confidence_masks
            m = masks[1] if len(masks) > 1 else masks[0]
            return np.array(m.numpy_view(), dtype=np.float32).reshape(rgb.shape[0], rgb.shape[1])
        res = self._seg.process(rgb)
        return np.asarray(res.segmentation_mask, dtype=np.float32)

    def close(self):
        for obj in (self._hands, self._seg):
            try:
                obj.close()
            except Exception:
                pass


class AndroidTasksBackend(VisionBackend):
    """MediaPipe Tasks (Java AAR) driven through pyjnius with mmap-based data transfer."""

    name = "mediapipe-tasks-android"

    def __init__(self, models_dir, work_dir):
        self.models_dir = models_dir
        self.work_dir = work_dir
        self._J = {}
        self._keep = []
        self._hands = self._seg = None
        self._hin = self._sin = self._mout = None
        self._hbmp = self._sbmp = None
        self._hshape = self._sshape = None
        self._ts_h = self._ts_s = 0

    # -- helpers ------------------------------------------------------------ #
    def _map_model(self, path):
        from jnius import autoclass
        raf = autoclass("java.io.RandomAccessFile")(path, "r")
        channel = raf.getChannel()
        buf = channel.map(autoclass("java.nio.channels.FileChannel$MapMode").READ_ONLY, 0, channel.size())
        self._keep.extend([raf, channel, buf])
        return buf

    def _new_bitmap(self, w, h):
        J = self._J
        return J["Bitmap"].createBitmap(int(w), int(h), J["Config"].ARGB_8888)

    def _upload(self, shared, bitmap, rgb):
        h, w = rgb.shape[:2]
        shared.u8[: h * w * 4].reshape(h, w, 4)[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        shared.jbuf.rewind()
        bitmap.copyPixelsFromBuffer(shared.jbuf)

    # -- lifecycle ---------------------------------------------------------- #
    def open(self):
        from jnius import autoclass
        A = autoclass
        J = self._J
        J["ctx"] = A("org.kivy.android.PythonActivity").mActivity
        J["BaseOptions"] = A("com.google.mediapipe.tasks.core.BaseOptions")
        J["RunningMode"] = A("com.google.mediapipe.tasks.vision.core.RunningMode")
        J["Bitmap"] = A("android.graphics.Bitmap")
        J["Config"] = A("android.graphics.Bitmap$Config")
        J["BitmapImageBuilder"] = A("com.google.mediapipe.framework.image.BitmapImageBuilder")
        J["Extractor"] = A("com.google.mediapipe.framework.image.ByteBufferExtractor")
        os.makedirs(self.work_dir, exist_ok=True)
        side = EngineConfig.ml_long_side
        nbytes = side * side * 4
        errors = []
        try:
            HL = A("com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker")
            HLO = A("com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker$HandLandmarkerOptions")
            base = (J["BaseOptions"].builder()
                    .setModelAssetBuffer(self._map_model(os.path.join(self.models_dir, "hand_landmarker.task")))
                    .build())
            opts = (HLO.builder().setBaseOptions(base).setRunningMode(J["RunningMode"].VIDEO)
                    .setNumHands(1).setMinHandDetectionConfidence(0.5)
                    .setMinHandPresenceConfidence(0.5).setMinTrackingConfidence(0.5).build())
            self._hands = HL.createFromOptions(J["ctx"], opts)
            self._hin = SharedBuffer(os.path.join(self.work_dir, "hand_in.rgba"), nbytes, java=True)
            self.has_hands = True
        except Exception as exc:
            log.exception("HandLandmarker init failed")
            errors.append("hands: %s" % exc)
        try:
            IS = A("com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter")
            ISO = A("com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter$ImageSegmenterOptions")
            base = (J["BaseOptions"].builder()
                    .setModelAssetBuffer(self._map_model(os.path.join(self.models_dir, "selfie_segmenter.tflite")))
                    .build())
            opts = (ISO.builder().setBaseOptions(base).setRunningMode(J["RunningMode"].VIDEO)
                    .setOutputConfidenceMasks(True).setOutputCategoryMask(False).build())
            self._seg = IS.createFromOptions(J["ctx"], opts)
            self._sin = SharedBuffer(os.path.join(self.work_dir, "seg_in.rgba"), nbytes, java=True)
            self._mout = SharedBuffer(os.path.join(self.work_dir, "seg_out.f32"), nbytes, java=True)
            self.has_seg = True
        except Exception as exc:
            log.exception("ImageSegmenter init failed")
            errors.append("segmentation: %s" % exc)
        self.reason = "; ".join(errors)

    def detect_hand(self, rgb, ts_ms):
        if self._hands is None:
            return None
        h, w = rgb.shape[:2]
        if self._hshape != (h, w):
            self._hbmp = self._new_bitmap(w, h)
            self._hshape = (h, w)
        self._ts_h = max(self._ts_h + 1, int(ts_ms))
        self._upload(self._hin, self._hbmp, rgb)
        image = self._J["BitmapImageBuilder"](self._hbmp).build()
        try:
            result = self._hands.detectForVideo(image, self._ts_h)
        finally:
            image.close()
        hands = result.landmarks()
        if hands.size() == 0:
            return None
        pts = hands.get(0)
        out = np.empty((21, 3), dtype=np.float32)
        for i in range(21):
            q = pts.get(i)
            out[i] = (q.x(), q.y(), q.z())
        return out

    def segment(self, rgb, ts_ms):
        if self._seg is None:
            return None
        h, w = rgb.shape[:2]
        if self._sshape != (h, w):
            self._sbmp = self._new_bitmap(w, h)
            self._sshape = (h, w)
        self._ts_s = max(self._ts_s + 1, int(ts_ms))
        self._upload(self._sin, self._sbmp, rgb)
        image = self._J["BitmapImageBuilder"](self._sbmp).build()
        try:
            result = self._seg.segmentForVideo(image, self._ts_s)
        finally:
            image.close()
        masks = result.confidenceMasks()
        if not masks.isPresent():
            return None
        lst = masks.get()
        n = lst.size()
        mask = lst.get(1 if n > 1 else 0)                       # [background, person]
        mw, mh = int(mask.getWidth()), int(mask.getHeight())
        src = self._J["Extractor"].extract(mask)
        need = mw * mh * 4
        if need > self._mout.nbytes:
            self._mout.close()
            self._mout = SharedBuffer(os.path.join(self.work_dir, "seg_out.f32"), need, java=True)
        src.rewind()
        self._mout.jbuf.rewind()
        self._mout.jbuf.put(src)
        arr = np.frombuffer(self._mout.mm, dtype="<f4", count=mw * mh).reshape(mh, mw).copy()
        for i in range(n):
            try:
                lst.get(i).close()
            except Exception:
                pass
        return arr

    def close(self):
        for obj in (self._hands, self._seg):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        for buf in (self._hin, self._sin, self._mout):
            try:
                if buf is not None:
                    buf.close()
            except Exception:
                pass
        self._keep = []


def create_backend(is_android, models_dir, work_dir):
    """Best available backend; never raises (falls back to the classic engine)."""
    try:
        backend = (AndroidTasksBackend(models_dir, work_dir) if is_android
                   else PythonMediaPipeBackend(models_dir))
        backend.open()
        if backend.has_hands or backend.has_seg:
            return backend
        return NullBackend(backend.reason or "no MediaPipe model could be loaded")
    except Exception as exc:
        log.exception("MediaPipe backend unavailable")
        return NullBackend(str(exc))


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #


def guided_filter(guide, src, radius, eps):
    """Edge-aware smoothing of `src` steered by gray `guide` (both float32, 0..1)."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    cov_ip = cv2.boxFilter(guide * src, -1, k) - mean_i * mean_p
    var_i = cv2.boxFilter(guide * guide, -1, k) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def _smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


class BackgroundModel:
    """Masked running average of the scene behind the user.

    Only pixels that are NOT person (dilated) are learned, so the person never
    leaks into the model and walking away reveals the real background.  Pixels never
    observed are filled by a coarse inpaint of their surroundings.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.acc = None
        self.full = None          # uint8 full resolution
        self.known = None         # float32 work resolution, >0.5 == observed
        self.small8 = None        # uint8 work resolution copy
        self.fill_small = None    # inpainted work resolution estimate
        self.gain = np.ones(3, np.float32)
        self.updates = 0
        self._fill_time = 0.0
        self._fill_busy = False
        self._k_excl = None
        self._shape = None

    @property
    def ready(self):
        return self.updates >= 2

    @property
    def coverage(self):
        return 0.0 if self.known is None else float((self.known > 0.5).mean())

    def _alloc(self, frame, small):
        h, w = frame.shape[:2]
        sh, sw = small.shape[:2]
        self.acc = np.zeros((h, w, 3), np.float32)
        self.full = np.zeros((h, w, 3), np.uint8)
        self.known = np.zeros((sh, sw), np.float32)
        self.small8 = np.zeros((sh, sw, 3), np.uint8)
        k = max(3, int(round(sw * 0.03)) | 1)
        self._k_excl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._shape = (h, w, sh, sw)

    def update(self, frame, small, person, moving=None):
        h, w = frame.shape[:2]
        sh, sw = small.shape[:2]
        if self._shape != (h, w, sh, sw):
            self.reset()
            self._alloc(frame, small)
        excl = cv2.dilate((person > 0.15).astype(np.uint8), self._k_excl)
        obs = excl == 0
        if moving is not None:
            obs &= (moving == 0)          # moving pixels are never learned (mask lag safety)
        new = obs & (self.known < 0.5)
        old = obs & ~new
        if new.any():
            m = cv2.resize(new.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.accumulateWeighted(frame, self.acc, 1.0, mask=m)
        if old.any():
            m = cv2.resize(old.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.accumulateWeighted(frame, self.acc, self.cfg.bg_learn, mask=m)
        if obs.any():
            self.known[obs] = np.minimum(1.0, self.known[obs] + self.cfg.bg_known_step)
            cv2.convertScaleAbs(self.acc, dst=self.full)
            self.small8 = cv2.resize(self.full, (sw, sh), interpolation=cv2.INTER_AREA)
            self.updates += 1

    def maybe_refresh_fill(self, now):
        """Start a background inpaint when unknown pixels exist (rate limited)."""
        if self.known is None or self._fill_busy or now - self._fill_time < 0.6:
            return
        hole = (self.known < 0.5)
        if not hole.any():
            self.fill_small = None
            return
        self._fill_busy = True
        self._fill_time = now
        img = self.small8.copy()
        hole8 = hole.astype(np.uint8) * 255

        def job():
            try:
                sh, sw = img.shape[:2]
                half = cv2.resize(img, (max(8, sw // 2), max(8, sh // 2)), interpolation=cv2.INTER_AREA)
                hm = cv2.resize(hole8, (half.shape[1], half.shape[0]), interpolation=cv2.INTER_NEAREST)
                hm = cv2.dilate(hm, np.ones((3, 3), np.uint8))
                done = cv2.inpaint(half, hm, 3, cv2.INPAINT_TELEA)
                self.fill_small = cv2.resize(done, (sw, sh), interpolation=cv2.INTER_LINEAR)
            except Exception:
                log.exception("inpaint failed")
            finally:
                self._fill_busy = False

        threading.Thread(target=job, name="bg-inpaint", daemon=True).start()


# --------------------------------------------------------------------------- #
# device benchmark (decides 1080p vs 720p before the camera opens)
# --------------------------------------------------------------------------- #


def _synthetic_fist():
    pts = np.zeros((21, 3), np.float32)
    for ang, base in zip((-15, -3, 9, 21), (5, 9, 13, 17)):
        th = np.deg2rad(ang)
        u = np.array([np.sin(th), np.cos(th)])
        for k, (r, z) in enumerate(((0.5, 0.0), (0.55, -0.2), (0.5, -0.25), (0.42, -0.15))):
            pts[base + k] = (u[0] * r, u[1] * r, z)
    out = np.empty_like(pts)
    out[:, 0] = 0.5 + pts[:, 0] * 0.3
    out[:, 1] = 0.8 - pts[:, 1] * 0.3
    out[:, 2] = pts[:, 2] * 0.3
    return out


def benchmark_pipeline(h, w, measure=8, limit=60):
    """Average CPU cost (ms) of one camera frame at portrait h x w on THIS device:
    RGBA -> RGB, rotate, full engine pass with compositing, preview conversion.
    Returns None when the measurement fails (callers then pick the safe resolution)."""
    try:
        sh, sw = InvisibilityEngine(NullBackend())._ml_shape(h, w)
        blob = np.zeros((sh, sw), np.float32)
        cv2.ellipse(blob, (sw // 2, sh // 2), (sw // 4, sh // 3), 0, 0, 360, 1.0, -1)
        fist = _synthetic_fist()

        class _Bench(VisionBackend):
            name = "bench"
            has_hands = True
            has_seg = True
            hand_on = False

            def detect_hand(self, rgb, ts_ms):
                return fist if self.hand_on else None

            def segment(self, rgb, ts_ms):
                return blob

        backend = _Bench()
        engine = InvisibilityEngine(backend)
        engine.start()
        rng = np.random.default_rng(1)
        raw = cv2.cvtColor(rng.integers(0, 255, (w, h, 3), dtype=np.uint8), cv2.COLOR_RGB2RGBA)   # landscape sensor frame
        times = []
        try:
            for _ in range(limit):
                t0 = time.perf_counter()
                rgb = cv2.rotate(cv2.cvtColor(raw, cv2.COLOR_RGBA2RGB), cv2.ROTATE_90_CLOCKWISE)
                out, info = engine.process(rgb)
                if h > 1280:
                    out = cv2.resize(out, (int(w * 1280 / h) // 2 * 2, 1280), interpolation=cv2.INTER_AREA)
                out[::-1].tobytes()
                dt = (time.perf_counter() - t0) * 1000.0
                if info["bg_ready"] and not backend.hand_on:
                    backend.hand_on = True                     # now raise the "fist": compositing starts
                elif info["progress"] > 0.001:
                    times.append(dt)
                if len(times) >= measure:
                    break
                time.sleep(0.004)
        finally:
            engine.stop()
        return float(np.mean(times)) if len(times) >= 3 else None
    except Exception:
        log.exception("benchmark failed")
        return None


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


class InvisibilityEngine:
    def __init__(self, backend, cfg=None):
        self.backend = backend
        self.cfg = cfg or EngineConfig()
        self.bg = BackgroundModel(self.cfg)
        self.tracker = GestureTracker(self.cfg)
        self.fps = FpsMeter()
        self.progress = 0.0
        self.manual_invisible = False
        self._hand_w = self._seg_w = None
        self._hand_seq = self._seg_seq = 0
        self._prob = None                 # latest raw person probability
        self._m_smooth = None
        self._have_mask = False
        self._idx = 0
        self._t_prev = None
        self._calib_until = 0.0
        self._move_hits = 0
        self._k_expand = None
        self._k_shape = None
        self._k_near = self._k_move = self._k_aug = None
        self._prev_small = None
        self._classic_k = None
        self.reset_reason = None
        self._rim = 0.0

    # -- lifecycle ---------------------------------------------------------- #
    def start(self):
        if self.backend.has_hands:
            self._hand_w = _Worker("hand-worker", self._hand_job)
            self._hand_w.start()
        if self.backend.has_seg:
            self._seg_w = _Worker("seg-worker", self._seg_job)
            self._seg_w.start()
        self._calib_until = time.monotonic() + self.cfg.calib_seconds
        log.info("engine started, backend=%s hands=%s seg=%s %s", self.backend.name,
                 self.backend.has_hands, self.backend.has_seg, self.backend.reason)

    def stop(self):
        for w in (self._hand_w, self._seg_w):
            if w is not None:
                w.stop()
        for w in (self._hand_w, self._seg_w):
            if w is not None:
                w.join(timeout=1.0)
        try:
            self.backend.close()
        except Exception:
            log.exception("backend close failed")

    def reset_background(self):
        self.bg.reset()
        self._m_smooth = None
        self._move_hits = 0
        self._calib_until = time.monotonic() + self.cfg.calib_seconds

    def reset_all(self):
        """Fresh start (camera switched): background, gesture state and fade."""
        self.reset_background()
        self.tracker.reset()
        self.progress = 0.0
        self.manual_invisible = False

    def set_manual(self, value):
        self.manual_invisible = bool(value)

    # -- workers ------------------------------------------------------------ #
    def _hand_job(self, item):
        rgb, ts = item
        lm = self.backend.detect_hand(rgb, ts)
        if lm is None:
            return "none"
        h, w = rgb.shape[:2]
        return classify_hand(lm, w, h)

    def _seg_job(self, item):
        rgb, ts = item
        return self.backend.segment(rgb, ts)

    # -- never leave the user with a dead app ----------------------------- #
    def _check_worker_health(self, now):
        """A model that never produced a single result (API mismatch, bad model file...) is
        switched off at runtime; the engine continues in classic mode with a manual toggle."""
        for attr, flag in (("_seg_w", "has_seg"), ("_hand_w", "has_hands")):
            w = getattr(self, attr)
            if w is not None and w.ok == 0 and w._errors >= 5:
                log.error("%s never succeeded - switching it off", w.name)
                setattr(self.backend, flag, False)
                note = "%s failed at runtime" % w.name
                self.backend.reason = (self.backend.reason + " | " + note).strip(" |")
                w.stop()
                setattr(self, attr, None)
                if flag == "has_seg":
                    self.bg.reset()
                    self._m_smooth = None
                    self._calib_until = now + self.cfg.calib_seconds

    # -- per frame ---------------------------------------------------------- #
    def _ml_shape(self, h, w):
        side = self.cfg.ml_long_side
        if h >= w:
            return side, max(32, int(round(w * side / h)))
        return max(32, int(round(h * side / w))), side

    def _refine(self, small, prob_new):
        sh, sw = small.shape[:2]
        if prob_new is not None:
            p = prob_new
            if p.shape != (sh, sw):
                p = cv2.resize(p, (sw, sh), interpolation=cv2.INTER_LINEAR)
            lo, hi = self.cfg.matte_lo, self.cfg.matte_hi
            m = _smoothstep(np.clip((p - lo) / (hi - lo), 0.0, 1.0)).astype(np.float32)
            if self._m_smooth is None or self._m_smooth.shape != m.shape:
                self._m_smooth = m
            else:
                wn = self.cfg.mask_new_weight
                self._m_smooth = (m * wn + self._m_smooth * (1.0 - wn)).astype(np.float32)
        if self._m_smooth is None:
            return np.zeros((sh, sw), np.float32)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) * (1.0 / 255.0)
        m = np.clip(guided_filter(gray, self._m_smooth, self.cfg.guided_radius, self.cfg.guided_eps), 0.0, 1.0)
        m = cv2.dilate(m, self._k_expand)
        return cv2.GaussianBlur(m, (0, 0), max(0.8, sw * self.cfg.feather_frac))

    def _ensure_kernels(self, sh, sw):
        if self._k_shape == (sh, sw):
            return
        ell = cv2.getStructuringElement
        r = max(1, int(round(sw * self.cfg.expand_frac)))
        self._k_expand = ell(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        rn = max(3, int(round(sw * 0.07)))
        self._k_near = ell(cv2.MORPH_ELLIPSE, (2 * rn + 1, 2 * rn + 1))
        self._k_move = ell(cv2.MORPH_ELLIPSE, (5, 5))
        self._k_aug = (ell(cv2.MORPH_ELLIPSE, (3, 3)), ell(cv2.MORPH_ELLIPSE, (7, 7)))
        self._k_shape = (sh, sw)

    def _augment(self, small, person):
        """Hybrid matte: grow the (slightly stale) ML matte where the live frame
        differs from the learned background - removes lag artefacts on fast motion."""
        g = self.bg.gain
        ref = cv2.multiply(self.bg.small8, (float(g[0]), float(g[1]), float(g[2]), 0.0), dtype=cv2.CV_8U)
        d = cv2.absdiff(small, ref).max(axis=2)
        cand = ((d > self.cfg.aug_thresh) & (self.bg.known > 0.5)).astype(np.uint8)
        near = cv2.dilate((person > 0.2).astype(np.uint8), self._k_near)
        cand &= near
        cand = cv2.morphologyEx(cand * 255, cv2.MORPH_OPEN, self._k_aug[0])
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, self._k_aug[1])
        aug = cv2.GaussianBlur(cand.astype(np.float32) * (1.0 / 255.0), (0, 0),
                               max(0.8, small.shape[1] * self.cfg.feather_frac))
        return np.maximum(person, aug)

    def _classic_person(self, small, now):
        """No segmentation model: difference to the learned background."""
        if not self.bg.ready or now < self._calib_until:
            return np.zeros(small.shape[:2], np.float32)
        diff = cv2.absdiff(small, self.bg.small8).max(axis=2)
        m = (diff > 30).astype(np.uint8) * 255
        if self._classic_k is None:
            self._classic_k = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._classic_k[0])
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._classic_k[1])
        return m.astype(np.float32) * (1.0 / 255.0)

    def _update_gain(self, small, person):
        valid = (self.bg.known > 0.5) & (person < 0.05)
        vs = valid[::2, ::2]
        if vs.sum() < 200:
            return
        cur = small[::2, ::2][vs].mean(axis=0)
        ref = self.bg.small8[::2, ::2][vs].mean(axis=0)
        g = np.clip(cur / (ref + 1.0), 0.6, 1.6).astype(np.float32)
        self.bg.gain = 0.85 * self.bg.gain + 0.15 * g

    def _check_camera_moved(self, small, person):
        """True after `move_steps` consecutive checks in which the live frame disagrees with the
        learned background.  The median is used so that local changes (a lagging matte edge, a
        TV in the background) cannot trigger it, while a moved camera - which changes most
        pixels - does."""
        valid = (self.bg.known > 0.5) & (person < 0.05)
        if valid.sum() < valid.size * 0.1:
            return False
        ref = (self.bg.small8.astype(np.float32) * self.bg.gain).clip(0, 255)
        d = np.abs(small.astype(np.float32) - ref).mean(axis=2)
        score = float(np.median(d[valid][::3]))
        self._move_hits = self._move_hits + 1 if score > self.cfg.move_thresh else 0
        return self._move_hits >= self.cfg.move_steps

    def _composite(self, frame, alpha, shim):
        h, w = frame.shape[:2]
        sh, sw = alpha.shape
        mask = alpha > 0.004
        if not mask.any():
            return frame
        ys = np.flatnonzero(mask.any(axis=1))
        xs = np.flatnonzero(mask.any(axis=0))
        y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
        sx, sy = w / sw, h / sh
        X0, X1 = max(0, int(x0 * sx)), min(w, int(np.ceil(x1 * sx)))
        Y0, Y1 = max(0, int(y0 * sy)), min(h, int(np.ceil(y1 * sy)))
        if X1 - X0 < 2 or Y1 - Y0 < 2:
            return frame
        size = (X1 - X0, Y1 - Y0)
        a = cv2.resize(alpha[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
        bg = self.bg.full[Y0:Y1, X0:X1]
        g = self.bg.gain
        if float(np.abs(g - 1.0).max()) > 0.02:
            bg = cv2.multiply(bg, (float(g[0]), float(g[1]), float(g[2]), 0.0), dtype=cv2.CV_8U)
        known = self.bg.known[y0:y1, x0:x1]
        fill = self.bg.fill_small
        roi = frame[Y0:Y1, X0:X1]
        if float(known.min()) < 0.95:                  # part of the area was never seen as background
            kn = np.clip((known - 0.2) / 0.5, 0.0, 1.0).astype(np.float32)
            kn = cv2.resize(kn, size, interpolation=cv2.INTER_LINEAR)
            if fill is not None:                       # coarse inpaint of the surroundings ...
                other = cv2.resize(fill[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
            else:                                      # ... or, until it is ready, keep the live pixels
                other = roi
            bg = cv2.blendLinear(np.ascontiguousarray(bg), np.ascontiguousarray(other), kn, 1.0 - kn)
        out = cv2.blendLinear(np.ascontiguousarray(roi), np.ascontiguousarray(bg), 1.0 - a, a)
        if shim > 0.02 and self.cfg.shimmer:
            rim = (4.0 * alpha[y0:y1, x0:x1] * (1.0 - alpha[y0:y1, x0:x1]))[..., None]
            tint = np.array(self.cfg.accent, np.float32) * (shim * 0.85)
            rim8 = cv2.resize((rim * tint).clip(0, 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
            out = cv2.add(out, rim8)
        frame[Y0:Y1, X0:X1] = out
        return frame

    def process(self, frame):
        """frame: contiguous writable uint8 RGB (H, W, 3). Returns (frame, info)."""
        now = time.monotonic()
        self.fps.tick(now)
        self._idx += 1
        h, w = frame.shape[:2]
        if self.bg.full is not None and self.bg.full.shape[:2] != (h, w):
            self.bg.reset()                       # camera resolution changed
            self._m_smooth = None
        sh, sw = self._ml_shape(h, w)
        self._ensure_kernels(sh, sw)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        ts = int(now * 1000)
        if self._hand_w is not None:
            self._hand_w.submit((small, ts))
        if self._seg_w is not None:
            self._seg_w.submit((small, ts))

        self._check_worker_health(now)

        # --- asynchronous results ---
        if self._hand_w is not None:
            res, seq = self._hand_w.latest()
            if seq != self._hand_seq:
                self._hand_seq = seq
                self.tracker.update(None if res in (None, "none") else res, now)
        prob_new = None
        if self._seg_w is not None:
            res, seq = self._seg_w.latest()
            if seq != self._seg_seq:
                self._seg_seq = seq
                if res is not None:
                    prob_new = res
                    self._have_mask = True
        target = self.tracker.invisible if self.backend.has_hands else self.manual_invisible

        # --- person matte ---
        if self.backend.has_seg:
            person = self._refine(small, prob_new)
        else:
            person = self._classic_person(small, now)

        # --- transition ---
        dt = 0.0 if self._t_prev is None else min(0.1, max(0.0, now - self._t_prev))
        self._t_prev = now
        step = dt / max(0.05, self.cfg.transition_s)
        if target and self.bg.ready:
            self.progress = min(1.0, self.progress + step)
        elif not target:
            self.progress = max(0.0, self.progress - step)

        # --- background learning (only while fully visible) ---
        reset_reason = None
        classic_learning = (not self.backend.has_seg) and now < self._calib_until
        can_learn = (self._have_mask or not self.backend.has_seg)
        if self.progress <= 0.001 and not target and can_learn and self._idx % self.cfg.bg_every == 0:
            moving = None
            if self._prev_small is not None and self._prev_small.shape == small.shape:
                dmove = cv2.absdiff(small, self._prev_small).max(axis=2)
                moving = cv2.dilate((dmove > self.cfg.move_gate).astype(np.uint8), self._k_move)
            suspicious = False
            if self.bg.ready and not classic_learning:
                # judge the live frame against the model BEFORE learning from it, and never learn
                # from frames that disagree with it (otherwise the model drifts to the new view)
                if self.backend.has_seg:
                    moved = self._check_camera_moved(small, person)
                else:                                 # classic mode: "everything is foreground" == moved
                    self._move_hits = self._move_hits + 1 if float(person.mean()) > 0.6 else 0
                    moved = self._move_hits >= self.cfg.move_steps
                if moved:
                    self.bg.reset()
                    self._m_smooth = None
                    self._move_hits = 0
                    self._calib_until = now + self.cfg.calib_seconds
                    reset_reason = "camera_moved"
                suspicious = self._move_hits > 0 or moved
            if not suspicious:
                self.bg.update(frame, small, np.zeros_like(person) if classic_learning else person, moving)
        self.reset_reason = reset_reason

        # --- composite ---
        eased = _smoothstep(self.progress)
        if self.progress > 0.001 and self.bg.ready:
            person_c = self._augment(small, person) if self.backend.has_seg else person
            self._update_gain(small, person_c)
            self.bg.maybe_refresh_fill(now)
            shim = float(np.sin(np.pi * self.progress)) if 0.0 < self.progress < 1.0 else 0.0
            frame = self._composite(frame, np.ascontiguousarray(person_c * eased, dtype=np.float32), shim)

        self._prev_small = small

        info = {
            "invisible": bool(target),
            "progress": float(self.progress),
            "hand": self.tracker.hand if self.backend.has_hands else "manual",
            "coverage": self.bg.coverage,
            "bg_ready": self.bg.ready,
            "backend": self.backend.name,
            "has_hands": self.backend.has_hands,
            "has_seg": self.backend.has_seg,
            "ml_fps": float(self._seg_w.fps.value) if self._seg_w is not None else 0.0,
            "reset": reset_reason,
            "calibrating": (not self.bg.ready) or (classic_learning),
        }
        return frame, info
