# -*- coding: utf-8 -*-
"""
persian.py - Persian / Arabic text preparation for Kivy.

Kivy's SDL2 text provider draws characters left-to-right and does no contextual
shaping, so Persian text would show up as disconnected, reversed letters.
`fa()` converts logical Persian text into
    1. contextual glyph forms (isolated / initial / medial / final, lam-alef
       ligatures) taken from the Unicode "Arabic Presentation Forms" blocks, and
    2. visual (right-to-left) character order, mirroring brackets and keeping
       numbers / Latin words readable,
so a plain LTR renderer displays it correctly.  The font must contain the
presentation forms (Vazirmatn, DejaVu Sans and the Android Noto Arabic fonts do).

The joining tables are derived from `unicodedata` at import time, so there are no
hand-typed code points and no third-party packages (arabic-reshaper / python-bidi)
to compile for Android.
"""
import re
import unicodedata

ZWNJ = "\u200c"
ZWJ = "\u200d"
TATWEEL = "\u0640"

_DIGITS_FA = {ord("0") + i: ord("\u06f0") + i for i in range(10)}
_DIGITS_FA.update({0x0660 + i: 0x06F0 + i for i in range(10)})
_NORMALIZE = {
    ord("\u064a"): "\u06cc",  # Arabic yeh   -> Persian yeh
    ord("\u0649"): "\u06cc",  # alef maksura -> Persian yeh
    ord("\u0643"): "\u06a9",  # Arabic kaf   -> Persian kaf
}
_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]")
_MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{",
           "<": ">", ">": "<", "\u00ab": "\u00bb", "\u00bb": "\u00ab"}
_KEYS = {"isolated": "iso", "final": "fin", "initial": "ini", "medial": "med"}


def _build_tables():
    forms, ligs = {}, {}
    for cp in list(range(0xFB50, 0xFE00)) + list(range(0xFE70, 0xFF00)):
        ch = chr(cp)
        dec = unicodedata.decomposition(ch)
        if not dec.startswith("<"):
            continue
        tag, _, rest = dec.partition(" ")
        key = _KEYS.get(tag.strip("<>"))
        if key is None:
            continue
        parts = [chr(int(x, 16)) for x in rest.split()]
        if len(parts) == 1:
            forms.setdefault(parts[0], {}).setdefault(key, ch)
        elif len(parts) == 2 and parts[0] == "\u0644":
            ligs.setdefault((parts[0], parts[1]), {}).setdefault(key, ch)
    forms[TATWEEL] = {"iso": TATWEEL, "fin": TATWEEL, "ini": TATWEEL, "med": TATWEEL}
    return forms, ligs


_FORMS, _LIGS = _build_tables()
_ALEFS = "\u0622\u0623\u0625\u0627"


def has_rtl(text):
    """True when `text` contains at least one right-to-left letter."""
    for ch in text:
        if unicodedata.bidirectional(ch) in ("R", "AL"):
            return True
    return False


def to_fa_digits(value):
    """Latin / Arabic-Indic digits -> Persian digits."""
    return str(value).translate(_DIGITS_FA)


def _next_joins(chars, j):
    return j < len(chars) and (chars[j] in _FORMS or chars[j] == ZWJ)


def shape(text):
    """Logical Persian text -> joined presentation forms (still logical order)."""
    chars = list(text)
    out = []
    prev_joins = False
    i, n = 0, len(chars)
    while i < n:
        ch = chars[i]
        if ch == "\u0644" and i + 1 < n and (ch, chars[i + 1]) in _LIGS and chars[i + 1] in _ALEFS:
            lig = _LIGS[(ch, chars[i + 1])]
            out.append(lig["fin"] if prev_joins and "fin" in lig else lig["iso"])
            prev_joins = False
            i += 2
            continue
        f = _FORMS.get(ch)
        if f is None:
            out.append(ch)
            prev_joins = ch == ZWJ
            i += 1
            continue
        dual = "ini" in f
        nxt = dual and _next_joins(chars, i + 1)
        if dual and nxt and prev_joins and "med" in f:
            form = "med"
        elif dual and nxt:
            form = "ini"
        elif prev_joins and "fin" in f:
            form = "fin"
        else:
            form = "iso"
        out.append(f.get(form, f["iso"]))
        prev_joins = bool(nxt)
        i += 1
    return "".join(c for c in out if c not in (ZWNJ, ZWJ))


def _visual_order(text):
    """Simplified UAX#9 (paragraph direction = RTL): returns the visual string."""
    n = len(text)
    if n == 0:
        return text
    cls = [unicodedata.bidirectional(c) for c in text]
    t = list(cls)
    last_strong = "R"                                # sot of an RTL paragraph
    for i in range(n):                               # W2: EN after an Arabic letter -> AN
        if t[i] in ("L", "R", "AL"):
            last_strong = t[i]
        elif t[i] == "EN" and last_strong == "AL":
            t[i] = "AN"
    w2 = list(t)
    for i in range(1, n - 1):                        # W4: 1 separator between 2 numbers
        if w2[i] in ("CS", "ES") and w2[i - 1] in ("EN", "AN") and w2[i + 1] == w2[i - 1]:
            if not (w2[i] == "ES" and w2[i - 1] == "AN"):
                t[i] = w2[i - 1]
    w4 = list(t)
    for i in range(n):                               # W5: terminators next to European numbers
        if w4[i] == "ET" and ((i > 0 and w4[i - 1] == "EN") or (i + 1 < n and w4[i + 1] == "EN")):
            t[i] = "EN"
    kind = []
    for c in t:
        if c in ("R", "AL"):
            kind.append("R")
        elif c == "L":
            kind.append("L")
        elif c in ("EN", "AN"):
            kind.append("N")
        else:
            kind.append(None)
    i = 0
    while i < n:                                     # N1 / N2 (numbers behave like R)
        if kind[i] is not None:
            i += 1
            continue
        j = i
        while j < n and kind[j] is None:
            j += 1
        left = "R" if i == 0 else ("R" if kind[i - 1] in ("R", "N") else "L")
        right = "R" if j == n else ("R" if kind[j] in ("R", "N") else "L")
        res = left if left == right else "R"
        for k in range(i, j):
            kind[k] = res
        i = j
    levels = [1 if k == "R" else 2 for k in kind]
    items = list(zip(text, levels))
    for lvl in range(max(levels), 0, -1):            # L2
        i = 0
        while i < n:
            if items[i][1] >= lvl:
                j = i
                while j < n and items[j][1] >= lvl:
                    j += 1
                items[i:j] = items[i:j][::-1]
                i = j
            else:
                i += 1
    return "".join(_MIRROR.get(c, c) if lv == 1 else c for c, lv in items)


def _fa_line(line, digits):
    line = _MARKS.sub("", line.translate(_NORMALIZE))
    if digits:
        line = line.translate(_DIGITS_FA)
    if not has_rtl(line):
        return line
    return _visual_order(shape(line))


def fa(text, digits=True):
    """Prepare (possibly multi-line) Persian text for a Kivy Label.

    Use halign='right' on labels that show it.  Wrap long paragraphs with
    `fa_wrap()` - Kivy's own wrapping would break the visual order.
    """
    if text is None:
        return ""
    return "\n".join(_fa_line(part, digits) for part in str(text).split("\n"))


def fa_wrap(text, max_chars=28, digits=True):
    """Greedy word wrap in logical order, then shape every line."""
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for word in para.split(" "):
            cand = word if not cur else cur + " " + word
            if len(cand) > max_chars and cur:
                lines.append(cur)
                cur = word
            else:
                cur = cand
        lines.append(cur)
    return "\n".join(_fa_line(l, digits) for l in lines)
