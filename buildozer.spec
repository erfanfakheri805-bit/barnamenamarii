[app]

# (str) نام برنامه روی گوشی / Title shown under the launcher icon
title = نامرئی‌ساز

# (str) شناسهٔ بسته = package.domain + "." + package.name  (برای کافه‌بازار/مایکت باید یکتا باشد)
package.name = invisibilityar
package.domain = ir.invisibilityar

source.dir = .
source.include_exts = py,png,jpg,kv,ttf,tflite,task,json,txt
source.exclude_dirs = tests, bin, .buildozer, .github, scripts, signing, __pycache__, venv, .git
source.exclude_patterns = *.md, requirements-desktop.txt, .gitignore, START_HERE.txt, COPY_OF_WORKFLOW_build.yml.txt

# (str) نسخهٔ برنامه. برای هر انتشار جدید در بازار افزایش دهید.
version = 1.0.0
android.numeric_version = 1

# ---------------------------------------------------------------------------
# requirements
#   * "mediapipe" عمداً اینجا نیست: بستهٔ پایتونی mediapipe برای اندروید وجود ندارد و
#     p4a برایش recipe ندارد، پس بیلد شکست می‌خورد. به‌جای آن کتابخانهٔ رسمی MediaPipe Tasks
#     (فایل AAR) از طریق android.gradle_dependencies به برنامه اضافه می‌شود و با pyjnius
#     صدا زده می‌شود (ar_engine.py ← AndroidTasksBackend).
#   * مدل‌ها (hand_landmarker.task و selfie_segmenter.tflite) در assets/models قرار می‌گیرند.
# ---------------------------------------------------------------------------
requirements = python3,kivy==2.3.0,numpy,opencv,pyjnius,android,plyer

orientation = portrait
fullscreen = 0

# ---------------------------------------------------------------------------
# permissions: دوربین، میکروفون (ضبط صدا)، ذخیره‌سازی (فقط اندروید ۹ و قدیمی‌تر واقعاً لازم است؛ در نسخه‌های جدید نادیده گرفته می‌شود)،
# دسترسی به ویدیوها (اندروید ۱۳+) و لرزش
# ---------------------------------------------------------------------------
android.permissions = CAMERA, RECORD_AUDIO, VIBRATE, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, READ_MEDIA_VIDEO

android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.archs = arm64-v8a, armeabi-v7a
android.accept_sdk_license = True

# AndroidX + کتابخانهٔ رسمی MediaPipe Tasks (دست و تقسیم‌بندی انسان)
android.enable_androidx = True
android.gradle_dependencies = com.google.mediapipe:tasks-vision:0.10.14

# apk یا aab — workflow گیت‌هاب هر دو را (یکی پس از دیگری) می‌سازد
android.release_artifact = apk
android.debug_artifact = apk

android.allow_backup = False
android.logcat_filters = *:S python:D

# آیکون و صفحهٔ شروع
icon.filename = %(source.dir)s/assets/icon.png
icon.adaptive_foreground.filename = %(source.dir)s/assets/icon_fg.png
icon.adaptive_background.filename = %(source.dir)s/assets/icon_bg.png
presplash.filename = %(source.dir)s/assets/presplash.png
android.presplash_color = #0F172A

# نسخهٔ پایدار python-for-android (سازگار با Kivy 2.3.0 / NDK 25b / API 33)
p4a.branch = v2024.01.21

[buildozer]
log_level = 2
warn_on_root = 1
