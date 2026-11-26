# -*- coding: utf-8 -*-
"""
recorder.py - records the composited AR frames (not the UI) to a gallery video.

Android
-------
* MediaRecorder with VideoSource.SURFACE (H.264 / MP4) + microphone audio (AAC).
* Frames are drawn onto the recorder Surface through a Bitmap.  The pixels reach Java
  via the file-backed shared memory of ar_engine.SharedBuffer (no pyjnius array
  conversion), so 720p/1080p at 30 FPS does not choke the bridge.
* Saved with MediaStore (RELATIVE_PATH = DCIM/InvisibilityAR) on Android 10+ - no storage
  permission needed - and with a plain file in /storage/emulated/0/DCIM/InvisibilityAR plus
  a media-scan on Android 9 and older.
* Frames are written on a fixed 30 FPS cadence by a dedicated thread, so the video has a
  constant frame rate even when the AR pipeline briefly stalls.

Desktop (development): cv2.VideoWriter ('mp4v'), video only, ~/Videos/InvisibilityAR.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("invisibility.recorder")

FOLDER = "InvisibilityAR"
MIN_SECONDS = 1.0           # MediaRecorder.stop() fails when nothing was encoded yet


class RecorderError(Exception):
    pass


def _even(v):
    return int(v) // 2 * 2


def _stamp():
    return time.strftime("Invisibility_%Y%m%d_%H%M%S")


class _CadenceRecorder:
    """Latest-frame mailbox + fixed-rate writer thread (subclasses implement I/O)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._thread = None
        self._halt = threading.Event()
        self.width = self.height = 0
        self.fps = 30
        self.frames = 0
        self.started_at = 0.0
        self.path = ""
        self.active = False
        self.error = None

    # to be provided by subclasses
    def _open(self, with_audio):
        raise NotImplementedError

    def _write(self, rgb):
        raise NotImplementedError

    def _close(self, ok):
        """Finalise; returns True when the file was kept."""
        raise NotImplementedError

    # -- public API ----------------------------------------------------------
    def start(self, width, height, fps=30, with_audio=True):
        self.width, self.height, self.fps = _even(width), _even(height), int(fps)
        self.frames = 0
        self.error = None
        self._halt.clear()
        self._open(with_audio)
        self.started_at = time.monotonic()
        self.active = True
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()
        return self.path

    def push(self, rgb):
        if self.active:
            with self._lock:
                self._frame = rgb

    @property
    def elapsed(self):
        return time.monotonic() - self.started_at if self.active else 0.0

    def stop(self):
        """Finalise the file. Blocking - call it from a worker thread."""
        if not self.active:
            return False, self.path
        wait = MIN_SECONDS - self.elapsed
        if wait > 0:
            time.sleep(wait)
        self.active = False
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        ok = self.frames > 3 and self.error is None
        try:
            kept = self._close(ok)
        except Exception as exc:                        # pragma: no cover - device specific
            log.exception("recorder close failed")
            self.error = exc
            kept = False
        return bool(kept), self.path

    # -- writer thread -------------------------------------------------------
    def _loop(self):
        period = 1.0 / self.fps
        nxt = time.monotonic()
        while not self._halt.is_set():
            now = time.monotonic()
            if now < nxt:
                time.sleep(min(0.004, nxt - now))
                continue
            nxt += period
            if now - nxt > 0.25:                        # fell far behind: resync
                nxt = now + period
            with self._lock:
                frame = self._frame
            if frame is None:
                continue
            try:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                self._write(frame)
                self.frames += 1
            except Exception as exc:
                log.exception("frame write failed")
                self.error = exc
                break


# --------------------------------------------------------------------------- #
class AndroidSurfaceRecorder(_CadenceRecorder):
    def __init__(self, work_dir):
        super().__init__()
        self.work_dir = work_dir
        self._mr = self._surface = self._bitmap = self._paint = None
        self._shared = None
        self._pfd = None
        self._uri = None
        self._legacy_path = None
        self._resolver = None
        self._hw = False
        self._J = {}

    def _bitrate(self):
        bits = int(self.width * self.height * self.fps * 0.25)
        return max(4_000_000, min(bits, 20_000_000))

    def _prepare_output(self, mr, A, ctx, sdk):
        name = _stamp() + ".mp4"
        if sdk >= 29:                                     # scoped storage: MediaStore
            ContentValues = A("android.content.ContentValues")
            Integer = A("java.lang.Integer")
            media = A("android.provider.MediaStore$Video$Media")
            self._resolver = ctx.getContentResolver()
            values = ContentValues()
            values.put("_display_name", name)
            values.put("mime_type", "video/mp4")
            values.put("relative_path", "DCIM/" + FOLDER)
            values.put("is_pending", Integer(1))
            self._uri = self._resolver.insert(media.EXTERNAL_CONTENT_URI, values)
            if self._uri is None:
                raise RecorderError("MediaStore insert failed")
            self._pfd = self._resolver.openFileDescriptor(self._uri, "rw")
            mr.setOutputFile(self._pfd.getFileDescriptor())
            self.path = "/storage/emulated/0/DCIM/%s/%s" % (FOLDER, name)
        else:                                             # legacy shared storage
            env = A("android.os.Environment")
            dcim = env.getExternalStoragePublicDirectory(env.DIRECTORY_DCIM).getAbsolutePath()
            folder = os.path.join(dcim, FOLDER)
            os.makedirs(folder, exist_ok=True)
            self._legacy_path = os.path.join(folder, name)
            mr.setOutputFile(self._legacy_path)
            self.path = self._legacy_path

    def _open(self, with_audio):
        from jnius import autoclass                       # noqa: WPS433 (Android only)
        from ar_engine import SharedBuffer
        A = autoclass
        ctx = A("org.kivy.android.PythonActivity").mActivity
        sdk = int(A("android.os.Build$VERSION").SDK_INT)
        self._hw = sdk >= 23
        MR = A("android.media.MediaRecorder")
        Src = A("android.media.MediaRecorder$VideoSource")
        ASrc = A("android.media.MediaRecorder$AudioSource")
        Fmt = A("android.media.MediaRecorder$OutputFormat")
        VEnc = A("android.media.MediaRecorder$VideoEncoder")
        AEnc = A("android.media.MediaRecorder$AudioEncoder")

        def build(audio):
            mr = MR(ctx) if sdk >= 31 else MR()
            if audio:
                mr.setAudioSource(ASrc.MIC)
            mr.setVideoSource(Src.SURFACE)
            mr.setOutputFormat(Fmt.MPEG_4)
            mr.setVideoEncoder(VEnc.H264)
            if audio:
                mr.setAudioEncoder(AEnc.AAC)
                mr.setAudioEncodingBitRate(128000)
                mr.setAudioSamplingRate(44100)
            mr.setVideoSize(self.width, self.height)
            mr.setVideoFrameRate(self.fps)
            mr.setVideoEncodingBitRate(self._bitrate())
            self._prepare_output(mr, A, ctx, sdk)
            mr.prepare()
            return mr

        try:
            try:
                self._mr = build(with_audio)
            except Exception:
                if not with_audio:
                    raise
                log.exception("audio recording unavailable - continuing without sound")
                self._discard_output()
                self._mr = build(False)
            self._surface = self._mr.getSurface()
            self._mr.start()
        except Exception as exc:
            self._cleanup(delete=True)
            raise RecorderError(str(exc)) from exc

        bitmap_cls = A("android.graphics.Bitmap")
        cfg = A("android.graphics.Bitmap$Config")
        self._bitmap = bitmap_cls.createBitmap(self.width, self.height, cfg.ARGB_8888)
        self._paint = A("android.graphics.Paint")()
        self._shared = SharedBuffer(os.path.join(self.work_dir, "rec_frame.rgba"),
                                    self.width * self.height * 4, java=True)

    def _write(self, rgb):
        h, w = self.height, self.width
        self._shared.u8[: w * h * 4].reshape(h, w, 4)[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        self._shared.jbuf.rewind()
        self._bitmap.copyPixelsFromBuffer(self._shared.jbuf)
        canvas = None
        if self._hw:
            try:
                canvas = self._surface.lockHardwareCanvas()
            except Exception:
                log.exception("hardware canvas unavailable - using the software canvas")
                self._hw = False
        if canvas is None:
            canvas = self._surface.lockCanvas(None)
        try:
            canvas.drawBitmap(self._bitmap, 0.0, 0.0, self._paint)
        finally:
            self._surface.unlockCanvasAndPost(canvas)

    # -- finishing -----------------------------------------------------------
    def _discard_output(self):
        try:
            if self._pfd is not None:
                self._pfd.close()
        except Exception:
            pass
        self._pfd = None
        try:
            if self._uri is not None and self._resolver is not None:
                self._resolver.delete(self._uri, None, None)
        except Exception:
            pass
        self._uri = None
        try:
            if self._legacy_path and os.path.exists(self._legacy_path):
                os.remove(self._legacy_path)
        except Exception:
            pass

    def _cleanup(self, delete):
        mr, self._mr = self._mr, None
        if mr is not None:
            try:
                mr.reset()
            except Exception:
                pass
            try:
                mr.release()
            except Exception:
                pass
        try:
            if self._surface is not None:
                self._surface.release()
        except Exception:
            pass
        self._surface = None
        if delete:
            self._discard_output()
        if self._shared is not None:
            self._shared.close()
            self._shared = None
        self._bitmap = None

    def _close(self, ok):
        from jnius import autoclass
        A = autoclass
        stopped = False
        try:
            if self._mr is not None:
                self._mr.stop()
                stopped = True
        except Exception:
            log.exception("MediaRecorder.stop failed (recording too short?)")
        keep = ok and stopped
        if keep:
            try:
                if self._pfd is not None:
                    self._pfd.close()
                    self._pfd = None
                if self._uri is not None:
                    values = A("android.content.ContentValues")()
                    values.put("is_pending", A("java.lang.Integer")(0))
                    self._resolver.update(self._uri, values, None, None)
                elif self._legacy_path:
                    ctx = A("org.kivy.android.PythonActivity").mActivity
                    A("android.media.MediaScannerConnection").scanFile(
                        ctx, [self._legacy_path], ["video/mp4"], None)
            except Exception:
                log.exception("publishing the video to the gallery failed")
        self._cleanup(delete=not keep)
        return keep


# --------------------------------------------------------------------------- #
class DesktopRecorder(_CadenceRecorder):
    def __init__(self, work_dir=None):
        super().__init__()
        self._vw = None

    def _open(self, with_audio):
        folder = os.path.join(os.path.expanduser("~"), "Videos", FOLDER)
        os.makedirs(folder, exist_ok=True)
        for ext, fourcc in (("mp4", "mp4v"), ("avi", "MJPG")):
            path = os.path.join(folder, _stamp() + "." + ext)
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), self.fps, (self.width, self.height))
            if vw.isOpened():
                self._vw, self.path = vw, path
                return
        raise RecorderError("cv2.VideoWriter could not be opened")

    def _write(self, rgb):
        self._vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def _close(self, ok):
        if self._vw is not None:
            self._vw.release()
            self._vw = None
        if not ok and self.path and os.path.exists(self.path):
            os.remove(self.path)
        return ok


def create_recorder(is_android, work_dir):
    return AndroidSurfaceRecorder(work_dir) if is_android else DesktopRecorder(work_dir)
