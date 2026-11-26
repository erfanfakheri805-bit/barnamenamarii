# -*- coding: utf-8 -*-
"""
Invisibility AR  (نامرئی‌ساز)  -  Kivy entry point.

Threads
-------
* Kivy main thread  : UI, GL texture upload, reads the camera texture (GL thread only).
* "pipeline" thread : orientation fix -> InvisibilityEngine.process() (OpenCV) -> recorder.
* MediaPipe workers : hand + segmentation (inside ar_engine).
* "recorder" thread : fixed 30 FPS writer (inside recorder.py).
The UI thread never runs OpenCV / MediaPipe; frames travel through latest-wins mailboxes.
"""
import json
import logging
import os
import sys
import threading
import time

os.environ.setdefault("KIVY_NO_ARGS", "1")

from kivy.config import Config                                       # noqa: E402
from kivy.utils import platform, get_color_from_hex                  # noqa: E402

IS_ANDROID = platform == "android"
if not IS_ANDROID:
    Config.set("graphics", "width", "414")
    Config.set("graphics", "height", "860")
    Config.set("graphics", "resizable", "0")
Config.set("kivy", "exit_on_escape", "0")

import cv2                                                           # noqa: E402
import numpy as np                                                   # noqa: E402
from kivy.animation import Animation                                 # noqa: E402
from kivy.app import App                                             # noqa: E402
from kivy.clock import Clock                                         # noqa: E402
from kivy.core.text import LabelBase                                 # noqa: E402
from kivy.core.window import Window                                  # noqa: E402
from kivy.graphics import Color, RoundedRectangle                    # noqa: E402
from kivy.graphics.texture import Texture                            # noqa: E402
from kivy.metrics import dp                                          # noqa: E402
from kivy.properties import (AliasProperty, BooleanProperty, ListProperty,     # noqa: E402
                             NumericProperty, ObjectProperty, StringProperty)
from kivy.uix.behaviors import ButtonBehavior                        # noqa: E402
from kivy.uix.floatlayout import FloatLayout                         # noqa: E402
from kivy.uix.label import Label                                     # noqa: E402
from kivy.uix.screenmanager import Screen                            # noqa: E402
from kivy.uix.widget import Widget                                   # noqa: E402

import ar_engine as ae                                               # noqa: E402
from persian import fa, fa_wrap, to_fa_digits                        # noqa: E402
from recorder import create_recorder                                 # noqa: E402

log = logging.getLogger("invisibility.app")
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def asset(*parts):
    return os.path.join(APP_DIR, "assets", *parts)


# --------------------------------------------------------------------------- #
# fonts - Vazirmatn (downloaded at build time), bundled DejaVu Sans as safety net
# --------------------------------------------------------------------------- #
def _first(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


_REG = _first([asset("fonts", "Vazirmatn-Regular.ttf"), asset("fonts", "DejaVuSans.ttf"),
               "/system/fonts/NotoNaskhArabic-Regular.ttf", "/system/fonts/NotoSansArabic-Regular.ttf",
               "/system/fonts/DroidSansArabic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
_BOLD = _first([asset("fonts", "Vazirmatn-Bold.ttf"), asset("fonts", "DejaVuSans-Bold.ttf"), _REG or ""]) or _REG
if _REG:
    LabelBase.register(name="Vazir", fn_regular=_REG, fn_bold=_BOLD or _REG)
    FONT_NAME = "Vazir"
else:                                                                # pragma: no cover
    FONT_NAME = "Roboto"
log.info("UI font: %s", _REG)

# --------------------------------------------------------------------------- #
# Persian strings ('~' marks a half-space / ZWNJ)
# --------------------------------------------------------------------------- #
STR = {
    "app_title": "نامرئی~ساز",
    "boot_perm": "درخواست مجوزهای دوربین و میکروفون…",
    "boot_models": "بارگذاری مدل~های هوش مصنوعی…",
    "boot_camera": "آماده~سازی دوربین…",
    "perm_denied": "برای کار کردن برنامه، اجازهٔ دسترسی به دوربین لازم است.",
    "retry": "تلاش دوباره",
    "open_settings": "تنظیمات برنامه",
    "fatal": "خطای غیرمنتظره رخ داد. برنامه را دوباره باز کنید.",
    "state_invisible": "نامرئی",
    "state_visible": "مرئی",
    "state_manual": "(دستی)",
    "hint_calibrate": "از جلوی دوربین کنار بروید تا پس~زمینه ثبت شود",
    "hint_ready": "مشت کنید تا نامرئی شوید؛ دست را باز کنید تا ظاهر شوید",
    "hint_manual": "حالت سازگار: برای نامرئی شدن دکمهٔ «نامرئی شو» را بزنید",
    "rec_saved": "ویدیو در گالری ذخیره شد",
    "rec_failed": "ضبط ویدیو انجام نشد",
    "rec_discarded": "ویدیو خیلی کوتاه بود و ذخیره نشد",
    "rec_no_audio": "میکروفون در دسترس نیست؛ بدون صدا ضبط می~شود",
    "bg_reset": "پس~زمینه دوباره ثبت می~شود؛ از جلوی دوربین کنار بروید",
    "cam_moved": "دوربین حرکت کرد؛ پس~زمینه دوباره ثبت می~شود",
    "quality_down": "برای روانی بیشتر، کیفیت به ۷۲۰p کاهش یافت",
    "backend_classic": "مدل هوش مصنوعی بارگذاری نشد؛ حالت سازگار فعال است",
    "orient_title": "جهت تصویر",
    "orient_rotate": "چرخش ۹۰ درجه",
    "orient_mirror": "آینه~ای (چپ و راست)",
    "orient_flip": "وارونگی عمودی",
    "close": "بستن",
    "orient_hint": "اگر تصویر کج یا وارونه است، از دکمهٔ تنظیمات (بالا) جهت را اصلاح کنید",
    "orient_q": "جهت تصویر درست است؟",
    "orient_yes": "بله، درست است",
    "orient_no": "نه، بچرخان",
    "manual_hide": "نامرئی شو",
    "manual_show": "ظاهر شو",
    "cam_fail": "دوربین باز نشد. برنامه را ببندید و دوباره باز کنید.",
}


def T(key, wrap=0):
    text = STR[key].replace("~", "\u200c")
    return fa_wrap(text, wrap) if wrap else fa(text)


# --------------------------------------------------------------------------- #
# small widgets (their canvas lives in invisibility.kv)
# --------------------------------------------------------------------------- #
class RoundedCard(FloatLayout):
    bg_color = ListProperty([0.12, 0.16, 0.23, 0.9])
    border_color = ListProperty([1, 1, 1, 0.07])
    radius = NumericProperty(dp(18))
    shadow = NumericProperty(1.0)


class IconButton(ButtonBehavior, FloatLayout):
    source = StringProperty("")
    bg_color = ListProperty([0.06, 0.09, 0.16, 0.0])
    ring_color = ListProperty([1, 1, 1, 0.0])
    icon_px = NumericProperty(dp(28))
    scale = NumericProperty(1.0)

    def on_press(self):
        Animation.cancel_all(self, "scale")
        Animation(scale=0.86, d=0.07, t="out_quad").start(self)

    def on_release(self):
        Animation.cancel_all(self, "scale")
        Animation(scale=1.0, d=0.18, t="out_back").start(self)


class ChipButton(ButtonBehavior, Label):
    pass


class AccentButton(ButtonBehavior, Label):
    pass


class StateChip(ButtonBehavior, RoundedCard):
    """Hand-state indicator; tapping it toggles the manual mode when no hand model is available."""


class RecordButton(ButtonBehavior, Widget):
    rec = NumericProperty(0.0)       # 0 idle -> 1 recording (circle morphs into a rounded square)
    ring = NumericProperty(0.0)      # pulse while recording
    scale = NumericProperty(1.0)
    _pulse = None

    def _get_inner(self):
        return (self.width - dp(20)) * (1 - 0.34 * self.rec)
    inner = AliasProperty(_get_inner, None, bind=("width", "rec"))

    def on_press(self):
        Animation.cancel_all(self, "scale")
        Animation(scale=0.9, d=0.07, t="out_quad").start(self)

    def on_release(self):
        Animation.cancel_all(self, "scale")
        Animation(scale=1.0, d=0.2, t="out_back").start(self)

    def set_recording(self, on):
        Animation.cancel_all(self, "rec")
        Animation(rec=1.0 if on else 0.0, d=0.28, t="out_cubic").start(self)
        if self._pulse is not None:
            self._pulse.cancel(self)
            self._pulse = None
        self.ring = 0.0
        if on:
            self._pulse = Animation(ring=1.0, d=1.0, t="out_quad") + Animation(ring=0.0, d=0.0)
            self._pulse.repeat = True
            self._pulse.start(self)


class LoadingRing(Widget):
    angle = NumericProperty(0.0)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._ev = Clock.schedule_interval(self._spin, 1 / 60.0)

    def _spin(self, dt):
        self.angle = (self.angle + 260 * dt) % 360

    def stop(self):
        if self._ev is not None:
            self._ev.cancel()
            self._ev = None


class PreviewView(Widget):
    """Draws the AR frame in a rounded rectangle (aspect-fit)."""
    texture = ObjectProperty(None, allownone=True)
    tex_ratio = NumericProperty(9 / 16.0)
    dim = NumericProperty(0.0)

    def __init__(self, **kw):
        super().__init__(**kw)
        with self.canvas:
            self._col = Color(0.09, 0.12, 0.2, 1)
            self._rect = RoundedRectangle(radius=[dp(24)])
            self._dim_col = Color(0.06, 0.09, 0.16, 0.0)
            self._dim_rect = RoundedRectangle(radius=[dp(24)])
        self.bind(pos=self._layout, size=self._layout, tex_ratio=self._layout,
                  texture=self._on_texture, dim=self._on_dim)

    def _layout(self, *_):
        W, H = self.size
        if W <= 0 or H <= 0:
            return
        r = max(0.2, self.tex_ratio)
        if W / H > r:
            h, w = H, H * r
        else:
            w, h = W, W / r
        pos = (self.x + (W - w) / 2.0, self.y + (H - h) / 2.0)
        for rect in (self._rect, self._dim_rect):
            rect.pos = pos
            rect.size = (w, h)

    def _on_texture(self, *_):
        self._rect.texture = self.texture
        self._col.rgba = (1, 1, 1, 1) if self.texture is not None else (0.09, 0.12, 0.2, 1)

    def _on_dim(self, *_):
        self._dim_col.a = self.dim


class LoadingScreen(Screen):
    pass


class CameraScreen(Screen):
    pass


# --------------------------------------------------------------------------- #
# camera sources
# --------------------------------------------------------------------------- #
class FrameMsg:
    __slots__ = ("data", "w", "h", "kind", "facing", "ts")

    def __init__(self, data, w, h, kind, facing, ts):
        self.data, self.w, self.h, self.kind, self.facing, self.ts = data, w, h, kind, facing, ts


class AndroidKivyCamera:
    """Kivy's SurfaceTexture based camera provider; frames are read back from the GL texture.

    No Java byte[] ever crosses pyjnius, so 1080p stays cheap.  Must be driven from the
    Kivy main thread.
    """

    def __init__(self, slot):
        self.slot = slot
        self._cam = None
        self._ev = None
        self._last_ts = None
        self._last_grab = 0.0
        self.facing = "back"
        self.sensor = 90
        self.size = (0, 0)

    @staticmethod
    def _pick_size(camera_cls, cid, target_long):
        cam = camera_cls.open(cid)
        try:
            sizes = cam.getParameters().getSupportedPreviewSizes()
            cand = [(int(sizes.get(i).width), int(sizes.get(i).height)) for i in range(sizes.size())]
        finally:
            cam.release()
        time.sleep(0.12)                                    # let the camera HAL settle
        wide = [c for c in cand if max(c) >= 1280] or cand

        def score(c):
            long_, short = max(c), min(c)
            return (abs(long_ / float(short) - 16 / 9.0) > 0.06, abs(long_ - target_long), -long_)
        w, h = min(wide, key=score)
        return (max(w, h), min(w, h))                       # landscape like the sensor

    def open(self, facing, target_long):
        from jnius import autoclass
        from kivy.core.camera import Camera as CoreCamera
        camera_cls = autoclass("android.hardware.Camera")
        info_cls = autoclass("android.hardware.Camera$CameraInfo")
        want = info_cls.CAMERA_FACING_FRONT if facing == "front" else info_cls.CAMERA_FACING_BACK
        cid, orientation = 0, 90
        for i in range(camera_cls.getNumberOfCameras()):
            info = info_cls()
            camera_cls.getCameraInfo(i, info)
            if info.facing == want:
                cid, orientation = i, int(info.orientation)
                break
        last_error = None
        for tl in [target_long] + [t for t in (1280, 960, 640) if t < target_long]:
            try:
                w, h = self._pick_size(camera_cls, cid, tl)
                log.info("opening camera %s (%s) %dx%d sensor=%d", cid, facing, w, h, orientation)
                self._cam = CoreCamera(index=cid, resolution=(w, h), stopped=True)
                self._tune()
                if not self._start_lean():
                    self._cam.start()
                    self._drop_preview_callback()
                self.facing, self.sensor, self.size = facing, orientation, (w, h)
                break
            except Exception as exc:                          # try a smaller size
                last_error = exc
                log.exception("camera open failed (target %s)", tl)
                self._release()
                time.sleep(0.25)
        else:
            raise RuntimeError("camera could not be opened: %s" % (last_error,))
        self._last_ts = None
        self._last_grab = time.monotonic()
        self._ev = Clock.schedule_interval(self._grab, 0)

    def _start_lean(self):
        """Start the preview exactly like Kivy's provider does (SurfaceTexture -> FBO, updated by a
        timer) but WITHOUT its NV21 callback buffers: pyjnius would convert megabytes per frame
        that we never use.  Falls back to the stock start() if the provider differs."""
        cam = self._cam
        try:
            cam.fps = 60.0                                    # latch new camera frames sooner
            cam.stopped = False
            cam._android_camera.startPreview()
            Clock.unschedule(cam._update)
            Clock.schedule_interval(cam._update, 1.0 / cam.fps)
            return True
        except Exception:
            log.exception("lean preview start unavailable - using Kivy's start()")
            return False

    def _drop_preview_callback(self):
        """Kivy's provider also requests NV21 byte[] callbacks; pyjnius would convert every
        1080p frame (megabytes) although only the GL texture is used - switch them off."""
        try:
            self._cam._android_camera.setPreviewCallbackWithBuffer(None)
        except Exception:
            log.exception("could not disable the preview callback")

    def _tune(self):
        try:
            cam = self._cam._android_camera
            params = cam.getParameters()
            ranges = params.getSupportedPreviewFpsRange()
            best = None
            for i in range(ranges.size()):
                r = ranges.get(i)
                lo, hi = int(r[0]), int(r[1])
                if hi >= 30000:
                    key = (abs(hi - 30000), -lo)
                    if best is None or key < best[0]:
                        best = (key, lo, hi)
            if best:
                params.setPreviewFpsRange(best[1], best[2])
            modes = params.getSupportedFocusModes()
            if modes is not None and modes.contains("continuous-video"):
                params.setFocusMode("continuous-video")
            params.setRecordingHint(True)
            cam.setParameters(params)
        except Exception:
            log.exception("camera tuning skipped")

    def _grab(self, dt):
        cam = self._cam
        if cam is None:
            return
        tex = cam.texture
        if tex is None:
            return
        now = time.monotonic()
        try:                                                # new frame? (SurfaceTexture timestamp)
            ts = cam._surface_texture.getTimestamp()
            if ts and ts == self._last_ts and now - self._last_grab < 0.5:
                return
            self._last_ts = ts
        except Exception:
            pass
        self._last_grab = now
        try:
            fbo = getattr(cam, "_fbo", None)
            pixels = fbo.pixels if fbo is not None else tex.pixels
        except Exception:
            pixels = tex.pixels
        w, h = tex.size
        self.slot.put(FrameMsg(pixels, int(w), int(h), "rgba", self.facing, now))

    def _release(self):
        cam, self._cam = self._cam, None
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                log.exception("camera stop failed")
            try:
                raw = getattr(cam, "_android_camera", None)
                if raw is not None:
                    raw.release()
            except Exception:
                pass

    def close(self):
        if self._ev is not None:
            self._ev.cancel()
            self._ev = None
        self._release()


class DesktopCamera:
    """Webcam for development on a PC (already upright, mirrored like a selfie)."""

    def __init__(self, slot):
        self.slot = slot
        self._cap = None
        self._halt = threading.Event()
        self.facing = "front"
        self.sensor = 0
        self.size = (1280, 720)

    def open(self, facing, target_long):
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._halt.clear()
        self.facing = facing
        threading.Thread(target=self._loop, name="webcam", daemon=True).start()

    def _loop(self):
        while not self._halt.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            if rgb.shape[1] > rgb.shape[0]:                 # phone-like portrait crop
                w = int(rgb.shape[0] * 9 / 16.0)
                x0 = (rgb.shape[1] - w) // 2
                rgb = np.ascontiguousarray(rgb[:, x0:x0 + w])
            self.slot.put(FrameMsg(rgb, rgb.shape[1], rgb.shape[0], "rgb", self.facing, time.monotonic()))
            time.sleep(0.005)

    def close(self):
        self._halt.set()
        time.sleep(0.05)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# --------------------------------------------------------------------------- #
# persisted settings (image orientation per camera)
# --------------------------------------------------------------------------- #
class OrientationStore:
    def __init__(self, path):
        self.path = path
        self.data = {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.data = json.load(fh)
        except Exception:
            self.data = {}

    def ensure(self, facing, sensor):
        if facing not in self.data:
            self.data[facing] = {"rot": int(sensor) % 360, "mirror": facing == "front", "vflip": False}

    def get(self, facing):
        return self.data.get(facing) or {"rot": 0, "mirror": False, "vflip": False}

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh)
        except Exception:
            log.exception("could not save settings")

    def rotate(self, facing):
        o = self.get(facing)
        o["rot"] = (int(o["rot"]) + 90) % 360
        self.data[facing] = o
        self.save()

    def toggle(self, facing, key):
        o = self.get(facing)
        o[key] = not o[key]
        self.data[facing] = o
        self.save()

    def apply(self, rgb, facing):
        o = self.get(facing)
        if o["vflip"]:
            rgb = cv2.flip(rgb, 0)
        rot = int(o["rot"]) % 360
        if rot == 90:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        elif rot == 270:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if o["mirror"]:
            rgb = cv2.flip(rgb, 1)
        return np.ascontiguousarray(rgb)


# --------------------------------------------------------------------------- #
# processing thread
# --------------------------------------------------------------------------- #
class Pipeline(threading.Thread):
    DISPLAY_MAX = 1280

    def __init__(self, orient):
        super().__init__(name="pipeline", daemon=True)
        self.orient = orient
        self.inbox = ae.LatestSlot()
        self.outbox = ae.LatestSlot()
        self.engine = None
        self.recorder = None
        self.fps = ae.FpsMeter()
        self.frame_size = (0, 0)

    def stop(self):
        self.inbox.close()

    def _to_rgb(self, msg):
        if msg.kind == "rgb":
            return msg.data
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.h, msg.w, 4)
        return self.orient.apply(cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB), msg.facing)

    def run(self):
        while not self.inbox.closed:
            msg = self.inbox.get(0.25)
            if msg is None:
                continue
            try:
                rgb = self._to_rgb(msg)
                if self.engine is not None:
                    out, info = self.engine.process(rgb)
                else:
                    out, info = rgb, {}
                h, w = out.shape[:2]
                self.frame_size = (w, h)
                rec = self.recorder
                if rec is not None:
                    rec.push(out)
                disp = out
                if max(h, w) > self.DISPLAY_MAX:
                    k = self.DISPLAY_MAX / float(max(h, w))
                    disp = cv2.resize(out, (int(w * k) // 2 * 2, int(h * k) // 2 * 2), interpolation=cv2.INTER_AREA)
                dh, dw = disp.shape[:2]
                info["fps"] = self.fps.tick()
                info["w"], info["h"] = w, h
                self.outbox.put((dw, dh, disp[::-1].tobytes(), info))     # bottom-up rows for GL
            except Exception:
                log.exception("pipeline error")
                time.sleep(0.05)


# --------------------------------------------------------------------------- #
# the application
# --------------------------------------------------------------------------- #
class InvisibilityApp(App):
    kv_file = os.path.join(APP_DIR, "invisibility.kv")
    title = "Invisibility AR"

    font_name = StringProperty(FONT_NAME)
    font_bold = StringProperty(FONT_NAME)
    c_bg = ListProperty(get_color_from_hex("#0F172A"))
    c_surface = ListProperty(get_color_from_hex("#1E293B"))
    c_accent = ListProperty(get_color_from_hex("#38BDF8"))
    c_text = ListProperty([1, 1, 1, 1])
    c_muted = ListProperty(get_color_from_hex("#94A3B8"))
    c_rec = ListProperty(get_color_from_hex("#F43F5E"))

    status_text = StringProperty("")
    error_mode = BooleanProperty(False)
    fps_text = StringProperty("")
    hint_text = StringProperty("")
    state_text = StringProperty("")
    state_icon = StringProperty("")
    state_color = ListProperty([0.06, 0.09, 0.16, 0.74])
    state_fg = ListProperty([1, 1, 1, 1])
    timer_text = StringProperty("")
    toast_text = StringProperty("")
    toast_opacity = NumericProperty(0.0)
    sheet_open = NumericProperty(0.0)
    recording = BooleanProperty(False)
    busy = BooleanProperty(False)
    orient_card_open = NumericProperty(0.0)
    diag_text = StringProperty("")
    classic_mode = BooleanProperty(False)
    manual_text = StringProperty("")

    # -- helpers used from KV ---------------------------------------------------
    def t(self, key):
        return T(key)

    def asset(self, *parts):
        return asset(*parts)

    # -- lifecycle ---------------------------------------------------------------
    def on_start(self):
        self.engine = None
        self.pipeline = None
        self.camera = (AndroidKivyCamera if IS_ANDROID else DesktopCamera)(None)
        self.facing = "back" if IS_ANDROID else "front"
        self.tier = 0
        self.target_long = 1920
        self.perm_audio = True
        self._tex = None
        self._switching = False
        self._last_hud = 0.0
        self._last_inv = None
        self._first_frame = False
        self._ready_since = None
        self._calib_start = time.monotonic()
        self._low_fps = 0
        self._cam_open_t = 0.0
        self._rec_t0 = 0.0
        self._toast_ev = None
        self._tick_ev = None
        self._gov_ev = None
        self._orient_k = 0
        self._classic = False
        self._timer_stamp = 0.0
        self.bench_ms = None
        self.manual_text = T("manual_hide")
        os.makedirs(self.user_data_dir, exist_ok=True)
        self.orient = OrientationStore(os.path.join(self.user_data_dir, "settings.json"))
        self.state_icon = asset("icons", "hand.png")
        self.state_text = T("state_visible")
        Window.bind(on_keyboard=self._on_key)
        if IS_ANDROID:
            self._keep_awake()
        Clock.schedule_once(self._begin_boot, 0.35)

    def on_pause(self):
        if self.recording:
            self._stop_record()
        self._close_camera()
        return True

    def on_resume(self):
        if self.engine is not None and self.root.current == "camera":
            self._open_camera()

    def on_stop(self):
        try:
            if self.recording:
                rec = self.pipeline.recorder if self.pipeline else None
                if rec is not None:
                    rec.stop()
        except Exception:
            pass
        self._close_camera()
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.engine is not None:
            self.engine.stop()

    @staticmethod
    def _keep_awake():
        try:
            from android.runnable import run_on_ui_thread
            from jnius import autoclass

            @run_on_ui_thread
            def flag():
                activity = autoclass("org.kivy.android.PythonActivity").mActivity
                params = autoclass("android.view.WindowManager$LayoutParams")
                activity.getWindow().addFlags(params.FLAG_KEEP_SCREEN_ON)
            flag()
        except Exception:
            log.exception("keep-screen-on failed")

    # -- boot sequence -----------------------------------------------------------
    def _set_status(self, key):
        self.status_text = T(key)

    def _begin_boot(self, dt=None):
        self.error_mode = False
        self._set_status("boot_perm")
        self._request_permissions(self._on_permissions)

    def retry_boot(self):
        self._teardown()
        self._begin_boot()

    def _teardown(self):
        for ev in (self._tick_ev, self._gov_ev):
            if ev is not None:
                ev.cancel()
        self._tick_ev = self._gov_ev = None
        self._close_camera()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        self._first_frame = False

    def _request_permissions(self, callback):
        if not IS_ANDROID:
            callback(True, True)
            return
        from android.permissions import Permission, check_permission, request_permissions
        from jnius import autoclass
        sdk = int(autoclass("android.os.Build$VERSION").SDK_INT)
        wanted = [Permission.CAMERA, Permission.RECORD_AUDIO]
        if sdk <= 28:                                        # scoped storage makes these unnecessary later on
            wanted += [Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE]

        def finished(perms=None, grants=None):
            cam = bool(check_permission(Permission.CAMERA))
            mic = bool(check_permission(Permission.RECORD_AUDIO))
            Clock.schedule_once(lambda dt: callback(cam, mic), 0)

        if all(check_permission(p) for p in wanted):
            finished()
        else:
            request_permissions(wanted, finished)

    def _on_permissions(self, camera_ok, audio_ok):
        self.perm_audio = audio_ok
        if not camera_ok:
            self.status_text = T("perm_denied")
            self.error_mode = True
            return
        self._set_status("boot_models")
        threading.Thread(target=self._init_engine, name="engine-init", daemon=True).start()

    def _choose_resolution(self):
        """1080p only when this phone's measured per-frame cost leaves headroom for 30 FPS."""
        self.bench_ms = None
        if not IS_ANDROID:
            self.tier, self.target_long = 1, 1280
            return
        ms = ae.benchmark_pipeline(1920, 1080)
        self.bench_ms = ms
        if ms is not None and ms <= 26.0:
            self.tier, self.target_long = 0, 1920
        else:
            self.tier, self.target_long = 1, 1280
        log.info("benchmark 1080p: %s ms -> %s", ms, "1080p" if self.tier == 0 else "720p")

    def _init_engine(self):
        try:
            self._choose_resolution()
            work = os.path.join(self.user_data_dir, "bridge")
            backend = ae.create_backend(IS_ANDROID, asset("models"), work)
            engine = ae.InvisibilityEngine(backend)
            engine.start()
            self.engine = engine
            Clock.schedule_once(lambda dt: self._engine_ready(backend), 0)
        except Exception:
            log.exception("engine init failed")
            Clock.schedule_once(lambda dt: self._fatal(), 0)

    def _fatal(self):
        self.status_text = T("fatal")
        self.error_mode = True

    def _engine_ready(self, backend):
        self._set_status("boot_camera")
        self.pipeline = Pipeline(self.orient)
        self.pipeline.engine = self.engine
        self.pipeline.start()
        self.camera.slot = self.pipeline.inbox
        self._classic = not (backend.has_hands and backend.has_seg)
        self.classic_mode = not backend.has_hands
        try:
            self._open_camera()
        except Exception:
            log.exception("camera open failed")
            self.status_text = T("cam_fail")
            self.error_mode = True
            return
        self._tick_ev = Clock.schedule_interval(self._ui_tick, 1 / 60.0)
        self._gov_ev = Clock.schedule_interval(self._governor, 1.0)

    # -- camera ------------------------------------------------------------------
    def _open_camera(self):
        self.camera.slot = self.pipeline.inbox
        self.camera.open(self.facing, self.target_long)
        self.orient.ensure(self.camera.facing, self.camera.sensor)
        self._cam_open_t = time.monotonic()
        self._update_diag()

    def _update_diag(self):
        parts = []
        backend = self.engine.backend if self.engine is not None else None
        if backend is not None:
            parts.append("engine: %s | hands: %s | seg: %s" % (
                backend.name, "ok" if backend.has_hands else "off", "ok" if backend.has_seg else "off"))
            if backend.reason:
                parts.append("note: " + backend.reason[:110])
        size = getattr(self.camera, "size", (0, 0))
        if size and size[0]:
            parts.append("camera: %dx%d | sensor %s | %s" % (size[0], size[1], self.camera.sensor, self.camera.facing))
        if self.bench_ms is not None:
            parts.append("benchmark 1080p: %.0f ms/frame" % self.bench_ms)
        self.diag_text = "\n".join(parts)

    def _close_camera(self):
        try:
            self.camera.close()
        except Exception:
            log.exception("camera close failed")

    def flip_camera(self):
        if self.recording or self.busy or self._switching or self.pipeline is None:
            return
        self._switching = True
        preview = self.root.get_screen("camera").ids.preview
        Animation(dim=0.88, d=0.16).start(preview)
        Clock.schedule_once(self._do_flip, 0.2)

    def _do_flip(self, dt):
        try:
            self._close_camera()
            self.facing = "front" if self.facing == "back" else "back"
            self.engine.reset_all()
            self._calib_start = time.monotonic()
            self._open_camera()
        except Exception:
            log.exception("camera flip failed")
        preview = self.root.get_screen("camera").ids.preview
        Animation(dim=0.0, d=0.3).start(preview)
        self._switching = False
        Clock.schedule_once(lambda dt: self._maybe_orient_card(), 1.2)

    def reset_background(self):
        if self.engine is not None:
            self.engine.reset_background()
            self._calib_start = time.monotonic()
            self.toast("bg_reset")

    def _governor(self, dt):
        """Adaptive quality: fall back from 1080p to 720p when the pipeline cannot keep up."""
        p = self.pipeline
        if p is None or self.tier == 1 or self.recording or self.busy or self._switching:
            return
        if time.monotonic() - self._cam_open_t < 6.0 or self.root.current != "camera":
            return
        fps = p.fps.value
        self._low_fps = self._low_fps + 1 if 0 < fps < 21 else 0
        if self._low_fps >= 5:
            self._low_fps = 0
            self.tier = 1
            self.target_long = 1280
            self._close_camera()
            self._open_camera()
            self._calib_start = time.monotonic()
            self.toast("quality_down")

    # -- UI tick -----------------------------------------------------------------
    def _upload(self, w, h, buf):
        preview = self.root.get_screen("camera").ids.preview
        tex = self._tex
        if tex is None or tuple(tex.size) != (w, h):
            tex = Texture.create(size=(w, h), colorfmt="rgb")
            tex.min_filter = "linear"
            tex.mag_filter = "linear"
            self._tex = tex
            preview.texture = tex
            preview.tex_ratio = w / float(h)
        tex.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
        preview.canvas.ask_update()

    def _ui_tick(self, dt):
        p = self.pipeline
        if p is None:
            return
        item = p.outbox.get_nowait()
        now = time.monotonic()
        if item is not None:
            w, h, buf, info = item
            self._upload(w, h, buf)
            if not self._first_frame:
                self._first_frame = True
                self._show_camera_screen()
            self._apply_info(info, now)
        if self.recording and now - getattr(self, "_timer_stamp", 0) > 0.25:
            self._timer_stamp = now
            sec = int(now - self._rec_t0)
            self.timer_text = to_fa_digits("%02d:%02d" % divmod(sec, 60))

    def _show_camera_screen(self):
        ring = self.root.get_screen("loading").ids.ring
        self.root.current = "camera"
        Clock.schedule_once(lambda dt: ring.stop(), 0.6)
        if self._classic:
            self.toast("backend_classic", 4.0)
        Clock.schedule_once(lambda dt: self._maybe_orient_card(), 1.6)

    # -- first-run orientation confirmation --------------------------------------
    def _maybe_orient_card(self):
        if self.orient.get(self.facing).get("ok"):
            return
        self._orient_k = 0
        Animation.cancel_all(self, "orient_card_open")
        Animation(orient_card_open=1.0, d=0.3).start(self)
        Clock.schedule_once(self._orient_card_timeout, 40.0)

    def _orient_card_timeout(self, dt):
        if self.orient_card_open > 0.5:
            self.orient_ok()

    def orient_ok(self):
        o = self.orient.get(self.facing)
        o["ok"] = True
        self.orient.data[self.facing] = o
        self.orient.save()
        Animation.cancel_all(self, "orient_card_open")
        Animation(orient_card_open=0.0, d=0.25).start(self)

    def orient_next(self):
        """'No' pressed: step through the 8 possible orientations until the picture is upright."""
        self._orient_k = (self._orient_k + 1) % 8
        o = self.orient.get(self.facing)
        o["rot"] = (int(self.camera.sensor) + 90 * (self._orient_k % 4)) % 360
        o["mirror"] = (self.facing == "front") != (self._orient_k >= 4)
        o["vflip"] = False
        self.orient.data[self.facing] = o
        self.orient.save()

    def _apply_info(self, info, now):
        if now - self._last_hud > 0.4:
            self._last_hud = now
            self.fps_text = to_fa_digits("%d FPS · %dp" % (round(info.get("fps", 0)), min(info["w"], info["h"])))
        inv = bool(info.get("invisible"))
        manual = not info.get("has_hands", True)
        self.classic_mode = manual                      # the manual toggle follows the live engine state
        key = (inv, manual)
        if key != self._last_inv:
            self._last_inv = key
            self._set_state(inv, manual)
        # hint chip
        cov = float(info.get("coverage", 0.0))
        if info.get("reset"):
            self._calib_start = now
        calibrating = (not info.get("bg_ready", False)) or bool(info.get("calibrating")) \
            or (cov < 0.85 and now - self._calib_start < 12.0)
        if calibrating:
            pct = int(round(cov * 100))
            self.hint_text = fa_wrap(STR["hint_calibrate"].replace("~", "\u200c") + " (%s٪)" % to_fa_digits(pct), 34)
            self._ready_since = None
        elif manual:
            self.hint_text = T("hint_manual")
        else:
            if self._ready_since is None:
                self._ready_since = now
            self.hint_text = fa_wrap(STR["hint_ready"].replace("~", "\u200c"), 34) if now - self._ready_since < 7 else ""
        if info.get("reset") == "camera_moved":
            self.toast("cam_moved")

    def _set_state(self, invisible, manual):
        text = T("state_invisible") if invisible else T("state_visible")
        if manual:
            text = fa(STR["state_manual"] + " " + ("نامرئی" if invisible else "مرئی"))
        self.state_text = text
        self.manual_text = T("manual_show") if invisible else T("manual_hide")
        self.state_icon = asset("icons", "fist.png" if invisible else "hand.png")
        if invisible:
            Animation(state_color=[0.22, 0.74, 0.97, 0.96], state_fg=[0.06, 0.09, 0.16, 1], d=0.28, t="out_cubic").start(self)
        else:
            Animation(state_color=[0.06, 0.09, 0.16, 0.74], state_fg=[1, 1, 1, 1], d=0.28, t="out_cubic").start(self)

    def toggle_manual(self):
        """Tap on the state chip: manual invisible toggle (only when no hand model is available)."""
        if self.engine is not None and not self.engine.backend.has_hands:
            self.engine.set_manual(not self.engine.manual_invisible)

    # -- toast / sheet -----------------------------------------------------------
    def toast(self, key, seconds=2.6):
        self.toast_text = fa_wrap(STR[key].replace("~", "\u200c"), 34)
        if self._toast_ev is not None:
            self._toast_ev.cancel()
        Animation.cancel_all(self, "toast_opacity")
        Animation(toast_opacity=1.0, d=0.18).start(self)
        self._toast_ev = Clock.schedule_once(
            lambda dt: Animation(toast_opacity=0.0, d=0.35).start(self), seconds)

    def toggle_sheet(self):
        target = 0.0 if self.sheet_open > 0.5 else 1.0
        Animation.cancel_all(self, "sheet_open")
        Animation(sheet_open=target, d=0.22, t="out_cubic").start(self)

    def orient_rotate(self):
        self.orient.rotate(self.facing)

    def orient_mirror(self):
        self.orient.toggle(self.facing, "mirror")

    def orient_flip(self):
        self.orient.toggle(self.facing, "vflip")

    # -- recording ---------------------------------------------------------------
    def toggle_record(self):
        if self.busy or self.pipeline is None or self._switching:
            return
        if self.recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self):
        w, h = self.pipeline.frame_size
        if w <= 0 or h <= 0:
            return
        k = min(1.0, 1280.0 / max(w, h))                     # 720p video (>= 720p, 30 FPS)
        w, h = int(w * k) // 2 * 2, int(h * k) // 2 * 2
        self.busy = True
        audio = self.perm_audio
        work = os.path.join(self.user_data_dir, "bridge")

        def job():
            rec, ok = None, False
            try:
                rec = create_recorder(IS_ANDROID, work)
                rec.start(w, h, 30, with_audio=audio)
                ok = True
            except Exception:
                log.exception("could not start recording")
            Clock.schedule_once(lambda dt: self._record_started(rec, ok), 0)
        threading.Thread(target=job, name="rec-start", daemon=True).start()

    def _record_started(self, rec, ok):
        self.busy = False
        if not ok:
            self.toast("rec_failed")
            return
        self.pipeline.recorder = rec
        self._rec_t0 = time.monotonic()
        self.timer_text = to_fa_digits("00:00")
        self.recording = True
        self.root.get_screen("camera").ids.rec_btn.set_recording(True)
        if not self.perm_audio:
            self.toast("rec_no_audio", 3.5)
        self._buzz()

    def _stop_record(self):
        rec = self.pipeline.recorder if self.pipeline else None
        if rec is None:
            return
        self.pipeline.recorder = None
        self.busy = True
        self.recording = False
        try:
            self.root.get_screen("camera").ids.rec_btn.set_recording(False)
        except Exception:
            pass

        def job():
            ok, path = False, ""
            try:
                ok, path = rec.stop()
            except Exception:
                log.exception("stop recording failed")
            Clock.schedule_once(lambda dt: self._record_stopped(ok, rec.frames), 0)
        threading.Thread(target=job, name="rec-stop", daemon=True).start()

    def _record_stopped(self, ok, frames):
        self.busy = False
        self.toast("rec_saved" if ok else ("rec_discarded" if frames <= 3 else "rec_failed"), 3.2)
        self._buzz()

    @staticmethod
    def _buzz():
        try:
            from plyer import vibrator
            vibrator.vibrate(0.035)
        except Exception:
            pass

    # -- misc --------------------------------------------------------------------
    def open_app_settings(self):
        if not IS_ANDROID:
            return
        try:
            from jnius import autoclass
            intent_cls = autoclass("android.content.Intent")
            settings = autoclass("android.provider.Settings")
            uri = autoclass("android.net.Uri")
            activity = autoclass("org.kivy.android.PythonActivity").mActivity
            intent = intent_cls(settings.ACTION_APPLICATION_DETAILS_SETTINGS)
            intent.setData(uri.fromParts("package", activity.getPackageName(), None))
            activity.startActivity(intent)
        except Exception:
            log.exception("cannot open app settings")

    def _on_key(self, window, key, *args):
        if key == 27:                                        # Android back button / Esc
            if self.sheet_open > 0.5:
                self.toggle_sheet()
                return True
            if self.recording:
                self._stop_record()
                return True
        return False


def _install_crash_log():
    def hook(exc_type, exc, tb):
        import traceback
        logging.getLogger("invisibility").critical("uncaught", exc_info=(exc_type, exc, tb))
        try:
            base = os.environ.get("ANDROID_PRIVATE") or os.path.expanduser("~")
            with open(os.path.join(base, "invisibility_crash.log"), "a", encoding="utf-8") as fh:
                fh.write("".join(traceback.format_exception(exc_type, exc, tb)) + "\n")
        except Exception:
            pass
    sys.excepthook = hook


if __name__ == "__main__":
    _install_crash_log()
    InvisibilityApp().run()
