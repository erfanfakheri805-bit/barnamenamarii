# -*- coding: utf-8 -*-
"""
Logic tests for main.py without Kivy installed:  python tests/test_app_logic.py

A tiny stub replaces the kivy modules so that main.py can be imported; then
* orientation transforms, the pipeline thread (Kivy-style RGBA frames -> engine -> outbox/recorder),
* HUD / state / toast logic and every Persian string,
* and invisibility.kv is cross-checked statically (indentation, app.* references, widget properties).
"""
import os
import re
import sys
import tempfile
import time
import types

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HOME"] = tempfile.mkdtemp()


# ----------------------------------------------------------------------- stub kivy
def install_stub():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class Dummy:
        def __init__(self, *a, **k):
            self.__dict__.update(k)

        def bind(self, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __getattr__(self, name):
            return Dummy()

        def __call__(self, *a, **k):
            return Dummy()

    class Anim:
        started = []

        def __init__(self, **k):
            self.k = k

        def __add__(self, other):
            return self

        def start(self, w):
            Anim.started.append(self.k)

        def cancel(self, w):
            pass

        @staticmethod
        def cancel_all(*a):
            pass

    class Ev:
        def cancel(self):
            pass

    class Clock:
        @staticmethod
        def schedule_once(fn, t=0):
            return Ev()

        @staticmethod
        def schedule_interval(fn, t):
            return Ev()

    class App:
        user_data_dir = tempfile.mkdtemp()

        def __init__(self, **k):
            pass

    class Behavior:
        def __init__(self, **k):
            pass

    def prop(default=None, *a, **k):
        if callable(default) and default in (list, str):
            return default()
        return default

    mod("kivy")
    mod("kivy.config", Config=Dummy())
    mod("kivy.utils", platform="linux", get_color_from_hex=lambda h: [0.1, 0.1, 0.1, 1.0])
    mod("kivy.animation", Animation=Anim)
    mod("kivy.app", App=App)
    mod("kivy.clock", Clock=Clock)
    mod("kivy.core")
    mod("kivy.core.text", LabelBase=Dummy())
    mod("kivy.core.window", Window=Dummy())
    mod("kivy.graphics", Color=Dummy, RoundedRectangle=Dummy)
    mod("kivy.graphics.texture", Texture=Dummy)
    mod("kivy.metrics", dp=lambda v: v)
    mod("kivy.properties", AliasProperty=lambda g, s=None, bind=(): property(g), BooleanProperty=prop,
        ListProperty=lambda d=None: list(d or []), NumericProperty=prop, ObjectProperty=prop, StringProperty=prop)
    mod("kivy.uix")
    mod("kivy.uix.behaviors", ButtonBehavior=Behavior)
    mod("kivy.uix.floatlayout", FloatLayout=Dummy)
    mod("kivy.uix.label", Label=Dummy)
    mod("kivy.uix.screenmanager", Screen=Dummy)
    mod("kivy.uix.widget", Widget=Dummy)
    return Anim


Anim = install_stub()
import main as M                                                         # noqa: E402
import ar_engine as ae                                                   # noqa: E402
import test_engine as T                                                  # noqa: E402
from persian import has_rtl                                              # noqa: E402


def test_strings():
    for key in M.STR:
        text = M.T(key)
        assert text and "~" not in text and "\u200c" not in text, key
        if has_rtl(M.STR[key]):
            assert any(0xFB50 <= ord(c) <= 0xFEFF for c in text), key
    wrapped = M.T("cam_moved", wrap=24)
    assert "\n" in wrapped
    print("  %d Persian strings shaped" % len(M.STR))


def test_orientation():
    store = M.OrientationStore(os.path.join(tempfile.mkdtemp(), "s.json"))
    store.ensure("back", 90)
    store.ensure("front", 270)
    assert store.get("back") == {"rot": 90, "mirror": False, "vflip": False}
    assert store.get("front")["mirror"] is True
    img = np.zeros((4, 6, 3), np.uint8)
    img[0, 0] = (255, 0, 0)                                  # top-left red
    out = store.apply(img, "back")                            # rotate 90 cw: (4,6) -> (6,4), top-left -> top-right
    assert out.shape == (6, 4, 3) and tuple(out[0, 3]) == (255, 0, 0) and out.flags["C_CONTIGUOUS"]
    store.rotate("back"); store.rotate("back"); store.rotate("back")
    assert store.get("back")["rot"] == 0 and store.apply(img, "back").shape == (4, 6, 3)
    store.toggle("back", "mirror")
    assert tuple(store.apply(img, "back")[0, 5]) == (255, 0, 0)
    reloaded = M.OrientationStore(store.path)
    assert reloaded.get("back")["mirror"] is True
    print("  orientation transforms + persistence ok")


def test_pipeline():
    store = M.OrientationStore(os.path.join(tempfile.mkdtemp(), "s.json"))
    store.ensure("back", 90)                                  # landscape sensor frames, rotate 90 cw
    backend = T.FakeBackend()
    eng = ae.InvisibilityEngine(backend)
    eng.start()
    p = M.Pipeline(store)
    p.engine = eng
    p.start()

    class Rec:
        n = 0

        def push(self, f):
            Rec.n += 1
            assert f.shape == (1280, 720, 3)

    p.recorder = Rec()
    bg = T.make_background(1280, 720)
    t0 = time.monotonic()
    got = None
    while time.monotonic() - t0 < 2.5:
        rgb_portrait = bg.copy()
        T.draw_person(rgb_portrait, 360, 640, 1280 / 1920.0)
        backend.feed(np.zeros((1280, 720), np.uint8), (270, 480))
        # simulate Kivy: landscape RGBA, rows bottom-up, sensor rotated 90 ccw relative to the portrait scene
        landscape = cv2.rotate(rgb_portrait, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgba = cv2.cvtColor(landscape, cv2.COLOR_RGB2RGBA)
        p.inbox.put(M.FrameMsg(rgba.tobytes(), rgba.shape[1], rgba.shape[0], "rgba", "back", time.monotonic()))
        item = p.outbox.get_nowait()
        if item is not None:
            got = item
        time.sleep(1 / 30.0)
    p.stop()
    eng.stop()
    assert got is not None, "pipeline produced nothing"
    w, h, buf, info = got
    assert (w, h) == (720, 1280) and len(buf) == w * h * 3
    assert info["fps"] > 5 and info["w"] == 720 and info["h"] == 1280
    # output rows are bottom-up for the GL texture: the first row of buf is the LAST image row
    disp = np.frombuffer(buf, np.uint8).reshape(h, w, 3)[::-1]
    assert np.abs(disp[5].astype(int) - bg[5].astype(int)).mean() < 3, "orientation of the output is wrong"
    assert Rec.n > 10
    print("  pipeline: %dx%d, %.0f fps, recorder got %d frames" % (w, h, info["fps"], Rec.n))


def test_hud_logic():
    app = M.InvisibilityApp.__new__(M.InvisibilityApp)
    defaults = dict(fps_text="", hint_text="", state_text="", state_icon="", state_color=[0] * 4, state_fg=[1] * 4,
                    toast_text="", toast_opacity=0.0, _last_hud=0.0, _last_inv=None, _ready_since=None,
                    _toast_ev=None, sheet_open=0.0, recording=False, timer_text="", _calib_start=0.0,
                    manual_text="")
    for k, v in defaults.items():
        setattr(app, k, v)
    info = {"fps": 29.6, "w": 720, "h": 1280, "invisible": False, "has_hands": True, "bg_ready": False,
            "calibrating": True, "coverage": 0.42, "reset": None}
    app._apply_info(info, 10.0)
    assert app.fps_text == "۳۰ FPS · ۷۲۰p", app.fps_text
    assert app.hint_text and "۴۲" in app.hint_text
    info.update(bg_ready=True, calibrating=False, coverage=1.0, invisible=True)
    app._apply_info(info, 11.0)
    assert app.state_icon.endswith("fist.png") and app.hint_text
    app._apply_info(info, 30.0)
    assert app.hint_text == "", "hint should fade after a few seconds"
    info.update(has_hands=False, invisible=False)
    app._apply_info(info, 31.0)
    assert app.state_icon.endswith("hand.png") and "دستی" not in app.state_text and app.hint_text
    app.toast("cam_moved")
    assert app.toast_text
    for key in M.STR:
        app.toast(key)
    print("  HUD, state chip and %d toasts ok" % len(M.STR))


def test_orientation_card_logic():
    app = M.InvisibilityApp.__new__(M.InvisibilityApp)
    app.orient = M.OrientationStore(os.path.join(tempfile.mkdtemp(), "s.json"))
    app.orient.ensure("back", 90)
    app.camera = types.SimpleNamespace(sensor=90)
    app.facing = "back"
    app._orient_k = 0
    app.orient_card_open = 0.0
    seen = []
    for _ in range(8):
        app.orient_next()
        o = app.orient.get("back")
        seen.append((o["rot"], o["mirror"], o["vflip"]))
    assert len(set(seen)) == 8 and seen[-1] == (90, False, False), seen      # all 8 orientations, then back to the start
    app.orient_ok()
    assert M.OrientationStore(app.orient.path).get("back")["ok"] is True
    app2 = M.InvisibilityApp.__new__(M.InvisibilityApp)
    app2._choose_resolution()
    assert (app2.tier, app2.target_long) == (1, 1280)                          # desktop: safe resolution
    print("  orientation confirmation cycle + resolution choice ok")


def _bare_app():
    app = M.InvisibilityApp.__new__(M.InvisibilityApp)
    defaults = dict(fps_text="", hint_text="", state_text="", state_icon="", state_color=[0] * 4, state_fg=[1] * 4,
                    toast_text="", toast_opacity=0.0, _last_hud=0.0, _last_inv=None, _ready_since=None,
                    _toast_ev=None, sheet_open=0.0, recording=False, timer_text="", _calib_start=0.0,
                    manual_text="", busy=False, status_text="", error_mode=False, classic_mode=False,
                    diag_text="", orient_card_open=0.0, _switching=False, perm_audio=True, _rec_t0=0.0,
                    tier=0, target_long=1920, _low_fps=0, _cam_open_t=0.0, bench_ms=None, engine=None,
                    pipeline=None, _orient_k=0, _classic=False, _tick_ev=None, _gov_ev=None, _first_frame=False,
                    facing="back", user_data_dir=tempfile.mkdtemp())
    for k, v in defaults.items():
        setattr(app, k, v)
    app.orient = M.OrientationStore(os.path.join(app.user_data_dir, "s.json"))
    return app


def test_record_flow():
    class Btn:
        def set_recording(self, on):
            self.on = on
    btn = Btn()
    app = _bare_app()
    app.root = types.SimpleNamespace(
        current="camera", get_screen=lambda n: types.SimpleNamespace(ids=types.SimpleNamespace(rec_btn=btn)))
    app.pipeline = types.SimpleNamespace(recorder=None, frame_size=(720, 1280))
    M.Clock.schedule_once = staticmethod(lambda fn, t=0: (fn(0), None)[1])      # run callbacks immediately
    app._start_record()
    for _ in range(60):
        if app.recording:
            break
        time.sleep(0.05)
    assert app.recording and app.pipeline.recorder is not None and btn.on is True
    rec = app.pipeline.recorder
    frame = np.full((1280, 720, 3), 90, np.uint8)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.4:
        app.pipeline.recorder.push(frame)
        time.sleep(1 / 30.0)
    app._stop_record()
    for _ in range(100):
        if not app.busy:
            break
        time.sleep(0.05)
    assert not app.busy and not app.recording and btn.on is False
    assert rec.path and os.path.getsize(rec.path) > 1000, "no video was written"
    assert app.toast_text == M.fa_wrap(M.STR["rec_saved"].replace("~", "\u200c"), 34)
    print("  record start/stop flow ok -> %d frames, %d KB" % (rec.frames, os.path.getsize(rec.path) // 1024))


def test_governor_and_teardown():
    class Cam:
        sensor, size, facing = 90, (1280, 720), "back"
        opened = closed = 0
        slot = None

        def open(self, facing, target):
            Cam.opened += 1
            self.target = target

        def close(self):
            Cam.closed += 1
    app = _bare_app()
    app.camera = Cam()
    app.root = types.SimpleNamespace(current="camera")
    fps = types.SimpleNamespace(value=12.0)
    app.pipeline = types.SimpleNamespace(fps=fps, inbox=object(), stop=lambda: None, recorder=None)
    app.engine = types.SimpleNamespace(backend=ae.NullBackend("x"), stop=lambda: None)
    app._cam_open_t = time.monotonic() - 60
    for _ in range(6):
        app._governor(0)
    assert app.tier == 1 and app.target_long == 1280 and Cam.opened == 1, (app.tier, Cam.opened)
    assert "classic" in app.diag_text and "camera: 1280x720" in app.diag_text, app.diag_text
    app._teardown()
    assert app.pipeline is None and app.engine is None and Cam.closed >= 2
    print("  governor (1080p -> 720p), diagnostics and teardown ok")


# ----------------------------------------------------------------------- static KV checks
def test_kv_static():
    kv = open(os.path.join(ROOT, "invisibility.kv"), encoding="utf-8").read()
    lines = kv.split("\n")
    assert "\t" not in kv, "tabs in kv"
    prev_indent = 0
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        assert indent % 4 == 0, "line %d: indent not a multiple of 4" % n
        assert indent <= prev_indent + 4, "line %d: indent jumps by more than one level" % n
        prev_indent = indent
    # app.<name> references must exist on the app class
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    body = src[src.index("class InvisibilityApp(App):"):src.index("def _install_crash_log")]
    members = set(re.findall(r"^    (\w+)\s*=", body, re.M)) | set(re.findall(r"^    def (\w+)\(", body, re.M))
    refs = set(re.findall(r"\bapp\.(\w+)", kv))
    missing = sorted(r for r in refs if r not in members)
    assert not missing, "kv uses app members that do not exist: %s" % missing
    # properties of custom widgets used in rules
    widget_props = {
        "RoundedCard": {"bg_color", "border_color", "radius", "shadow"},
        "IconButton": {"source", "bg_color", "ring_color", "icon_px", "scale"},
        "RecordButton": {"rec", "ring", "scale", "inner"},
        "LoadingRing": {"angle"},
    }
    for cls, props in widget_props.items():
        cbody = src[src.index("class %s(" % cls):]
        cbody = cbody[:cbody.index("\nclass ", 10)] if "\nclass " in cbody[10:] else cbody
        for p in props:
            assert re.search(r"^\s+%s\s*=" % p, cbody, re.M), "%s lacks property %s" % (cls, p)
    for cls in ("RoundedCard", "IconButton", "ChipButton", "AccentButton", "StateChip", "RecordButton", "LoadingRing",
                "PreviewView", "LoadingScreen", "CameraScreen"):
        assert re.search(r"^class %s\(" % cls, src, re.M), "class %s missing in main.py" % cls
        assert re.search(r"^\s*(<%s>:|%s:)" % (cls, cls), kv, re.M) or cls in ("ChipButton",), cls
    for key in re.findall(r"app\.t\('(\w+)'\)", kv):
        assert key in M.STR, "unknown string key in kv: " + key
    for rel in re.findall(r"app\.asset\(([^)]*)\)", kv):
        parts = [x.strip().strip("'\"") for x in rel.split(",")]
        assert os.path.isfile(os.path.join(ROOT, "assets", *parts)), "missing asset " + rel
    print("  kv: indentation, %d app.* references, widget properties, string keys and assets ok" % len(refs))


if __name__ == "__main__":
    for name, fn in (("persian strings", test_strings), ("orientation", test_orientation),
                     ("pipeline thread", test_pipeline), ("hud logic", test_hud_logic),
                     ("orientation card", test_orientation_card_logic),
                     ("record flow", test_record_flow), ("governor + teardown", test_governor_and_teardown),
                     ("kv static checks", test_kv_static)):
        print(name)
        fn()
    print("APP LOGIC TESTS PASSED")
