"""Render the pottery-hobby Shorts video (1080x1920, 30fps).

usage: python3 render.py <narration_dir> <out.mp4> [--preview SCENE_INDEX]
"""
import json, math, os, subprocess, sys, zlib
from contextlib import contextmanager

import cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "assets", "media")
FONTS = os.path.join(HERE, "assets", "fonts")
W, H, FPS = 1080, 1920, 30
BOIL = 3                      # redraw wobble every 3 frames (= "on threes" hand-drawn feel)

INK = (0.22, 0.16, 0.12)
WHITE = (1, 1, 1)
CLAY = (0.86, 0.80, 0.71)
CLAY_D = (0.70, 0.62, 0.52)
SKIN = (0.99, 0.86, 0.76)
ORANGE = (0.98, 0.55, 0.20)
YELLOW = (1.0, 0.86, 0.32)
RED = (0.86, 0.27, 0.22)
BLUE = (0.45, 0.70, 0.92)
GREY = (0.72, 0.74, 0.77)
PINK = (0.96, 0.55, 0.62)
SAGE = (0.62, 0.72, 0.60)

PEN = os.path.join(FONTS, "NanumPenScript_400Regular.ttf")
GAEGU = os.path.join(FONTS, "Gaegu_700Bold.ttf")


# ----------------------------------------------------------------------------- utils
def rng(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, t0, t1):
    return clamp((t - t0) / max(1e-6, t1 - t0))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out_back(x):
    x = clamp(x)
    c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def ell(cx, cy, rx, ry, a0=0.0, a1=2 * math.pi, n=72):
    a = np.linspace(a0, a1, n)
    return np.c_[cx + rx * np.cos(a), cy + ry * np.sin(a)]


def qbez(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def cbez(p0, p1, p2, p3, n=30):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def resample(pts, step=5.0):
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.r_[0, np.cumsum(L)]
    n = max(2, int(cum[-1] / step))
    tt = np.linspace(0, cum[-1], n)
    return np.c_[np.interp(tt, cum, pts[:, 0]), np.interp(tt, cum, pts[:, 1])]


def wobble(pts, amp, key):
    p = resample(pts)
    n = len(p)
    r = rng(*key)
    k = max(3, n // 14 + 3)
    ctrl = r.normal(0, amp, (k, 2))
    tt = np.linspace(0, k - 1, n)
    off = np.c_[np.interp(tt, np.arange(k), ctrl[:, 0]), np.interp(tt, np.arange(k), ctrl[:, 1])]
    return p + off


# ----------------------------------------------------------------------------- canvas
class Canvas:
    """RGBA PIL frame + helpers to draw cairo vector layers and PIL text on it."""

    def __init__(self, base, frame):
        self.img = base.convert("RGBA") if base.mode != "RGBA" else base.copy()
        self.f = frame
        self.b = frame // BOIL

    @contextmanager
    def vec(self):
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx = cairo.Context(surf)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        yield ctx
        surf.flush()
        layer = Image.frombuffer("RGBA", (W, H), bytes(surf.get_data()), "raw", "BGRa", 0, 1)
        self.img.alpha_composite(layer)

    # --- vector primitives (all take a cairo ctx)
    def stroke(self, ctx, pts, w=7, col=INK, p=1.0, key=(0,), amp=2.2, alpha=1.0,
               closed=False, sketch=True, outline=None):
        if p <= 0:
            return
        pts = np.asarray(pts, float)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        q = wobble(pts, amp, key + (self.b,))
        n = max(2, int(math.ceil(len(q) * clamp(p))))
        q = q[:n]
        if outline is not None:   # dark under-stroke so white ink reads on photos
            self._poly(ctx, q)
            ctx.set_source_rgba(*outline, 0.55 * alpha)
            ctx.set_line_width(w + 7)
            ctx.stroke()
        self._poly(ctx, q)
        ctx.set_source_rgba(*col, alpha)
        ctx.set_line_width(w)
        ctx.stroke()
        if sketch:
            q2 = wobble(pts, amp * 1.6, key + ("s", self.b))[:n]
            self._poly(ctx, q2)
            ctx.set_source_rgba(*col, 0.35 * alpha)
            ctx.set_line_width(max(1.5, w * 0.4))
            ctx.stroke()

    def fill(self, ctx, pts, col, key=(0,), amp=3.0, alpha=0.92):
        q = wobble(np.vstack([pts, pts[:1]]), amp, key + ("f", self.b))
        self._poly(ctx, q)
        ctx.close_path()
        ctx.set_source_rgba(*col, alpha)
        ctx.fill()

    def shape(self, ctx, pts, col, key, w=7, p=1.0, fill_alpha=0.92, ink=INK):
        """Watercolour-ish fill (fades in after outline) + wobbly outline."""
        if p > 0.6:
            self.fill(ctx, pts, col, key, alpha=fill_alpha * ease((p - 0.6) / 0.4))
        self.stroke(ctx, pts, w, ink, p=min(1, p / 0.6) if p < 1 else 1, key=key, closed=True)

    @staticmethod
    def _poly(ctx, q):
        ctx.new_path()
        ctx.move_to(*q[0])
        for x, y in q[1:]:
            ctx.line_to(x, y)

    # --- text
    def text(self, s, x, y, size=90, font=PEN, col=(56, 41, 31), stroke=0, stroke_col=(255, 255, 255),
             p=1.0, rot=0.0, jitter=1.5, scale=1.0, alpha=1.0, key=0):
        if p <= 0 or alpha <= 0 or scale <= 0.01:
            return
        n = max(1, int(round(len(s) * clamp(p))))
        s2 = s[:n]
        r = rng("t", key, s, self.b)
        jx, jy = r.normal(0, jitter, 2)
        jr = r.normal(0, 0.8) if jitter else 0
        im = text_img(s2, int(size * scale), font, col, stroke, stroke_col)
        if alpha < 1:
            a = im.getchannel("A").point(lambda v: int(v * alpha))
            im = im.copy()
            im.putalpha(a)
        if rot + jr:
            im = im.rotate(rot + jr, resample=Image.BICUBIC, expand=True)
        # anchor: centre of full string (so typing doesn't shift it)
        full = text_img(s, int(size * scale), font, col, stroke, stroke_col)
        x0 = int(x - full.width / 2 + jx)
        y0 = int(y - im.height / 2 + jy)
        if rot + jr:
            x0 = int(x - im.width / 2 + jx) if n == len(s) else x0
        _paste(self.img, im, x0, y0)


def _paste(dst, im, x, y):
    # alpha_composite with clipping for negative / overflowing offsets
    sx, sy = max(0, -x), max(0, -y)
    x, y = max(0, x), max(0, y)
    w = min(im.width - sx, W - x)
    h = min(im.height - sy, H - y)
    if w <= 0 or h <= 0:
        return
    dst.alpha_composite(im.crop((sx, sy, sx + w, sy + h)), (x, y))


_TCACHE = {}


def text_img(s, size, font, col, stroke, stroke_col):
    k = (s, size, font, col, stroke, stroke_col)
    if k in _TCACHE:
        return _TCACHE[k]
    ft = ImageFont.truetype(font, size)
    lines = s.split("\n")
    lh = int(size * 1.12)
    widths = [ft.getbbox(l or " ", stroke_width=stroke)[2] for l in lines]
    w = max(widths) + 2 * stroke + 20
    h = lh * len(lines) + 2 * stroke + 20
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        d.text((w / 2, 10 + stroke + i * lh + lh / 2), l, font=ft, fill=col + (255,), anchor="mm",
               stroke_width=stroke, stroke_fill=stroke_col + (255,))
    _TCACHE[k] = im
    return im


# ----------------------------------------------------------------------------- backgrounds
def make_paper(seed):
    r = np.random.default_rng(seed)
    lf = Image.fromarray((r.normal(0.5, 0.18, (H // 24, W // 24)).clip(0, 1) * 255).astype(np.uint8))
    lf = np.asarray(lf.resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(30)), float) / 255 - 0.5
    fine = r.normal(0, 3.2, (H, W))
    base = np.array([246, 239, 227], float)
    img = base[None, None, :] + (lf * 16)[..., None] * np.array([1, 1, 1.25]) + fine[..., None]
    yy, xx = np.mgrid[0:H, 0:W]
    v = ((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2
    img *= (1 - 0.22 * v)[..., None]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


PAPERS = [make_paper(i) for i in range(3)]


def paper(frame):
    return PAPERS[(frame // BOIL) % 3]


def load(name):
    return ImageOps.exif_transpose(Image.open(os.path.join(MEDIA, name))).convert("RGB")


class Photo:
    """Full-bleed photo with ken-burns zoom and normalised-coordinate mapping for doodles."""

    def __init__(self, name, z0=1.0, z1=1.07, c0=(0.5, 0.5), c1=(0.5, 0.5)):
        self.im = load(name)
        self.z0, self.z1, self.c0, self.c1 = z0, z1, c0, c1
        self.sk = None

    def _tf(self, u):
        iw, ih = self.im.size
        z = self.z0 + (self.z1 - self.z0) * u
        s = max(W / iw, H / ih) * z
        cx = (self.c0[0] + (self.c1[0] - self.c0[0]) * u) * iw
        cy = (self.c0[1] + (self.c1[1] - self.c0[1]) * u) * ih
        cx = clamp(cx, W / 2 / s, iw - W / 2 / s)
        cy = clamp(cy, H / 2 / s, ih - H / 2 / s)
        return s, cx, cy

    def frame(self, u, src=None):
        s, cx, cy = self._tf(u)
        box = (cx - W / 2 / s, cy - H / 2 / s, cx + W / 2 / s, cy + H / 2 / s)
        return (src or self.im).resize((W, H), Image.BICUBIC, box=box)

    def pt(self, u, nx, ny):
        s, cx, cy = self._tf(u)
        iw, ih = self.im.size
        return ((nx * iw - cx) * s + W / 2, (ny * ih - cy) * s + H / 2)

    def sketch(self):
        if self.sk is None:
            self.sk = sketchify(self.im)
        return self.sk


def sketchify(im):
    g = np.asarray(im.convert("L"), float)
    inv = Image.fromarray((255 - g).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(4, im.width // 140)))
    b = np.asarray(inv, float)
    dodge = np.clip(g * 255 / (256 - b), 0, 255)
    dodge = 255 - (255 - dodge) * 1.6
    d = np.clip(dodge, 0, 255) / 255
    pap = np.array([246, 239, 227], float)
    out = pap[None, None, :] * d[..., None] + np.array([70, 52, 40]) * (1 - d[..., None])
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


class Video:
    def __init__(self, name, start=0.0):
        self.path = os.path.join(MEDIA, name)
        self.start = start
        self.p = None
        self.last = None

    def next(self):
        if self.p is None:
            vf = f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
            self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(self.start), "-i", self.path,
                                       "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                      stdout=subprocess.PIPE)
        buf = self.p.stdout.read(W * H * 3)
        if len(buf) == W * H * 3:
            self.last = Image.frombuffer("RGB", (W, H), buf)
        return self.last


def polaroid(im, width, angle, caption_h=150):
    pw = width
    ph_img = int(im.height * (pw - 60) / im.width)
    card = Image.new("RGBA", (pw, ph_img + 30 + caption_h), (252, 250, 244, 255))
    card.paste(im.resize((pw - 60, ph_img), Image.LANCZOS), (30, 30))
    sh = Image.new("RGBA", (card.width + 80, card.height + 80), (0, 0, 0, 0))
    sh.paste((40, 30, 20, 90), (50, 55, 50 + card.width, 55 + card.height))
    sh = sh.filter(ImageFilter.GaussianBlur(16))
    sh.alpha_composite(card, (40, 40))
    # masking tape
    tape = Image.new("RGBA", (230, 70), (236, 222, 180, 190))
    td = ImageDraw.Draw(tape)
    for i in range(0, 230, 14):
        td.polygon([(i, 0), (i + 7, 0), (i + 3, 6)], fill=(0, 0, 0, 0))
        td.polygon([(i, 70), (i + 7, 70), (i + 3, 64)], fill=(0, 0, 0, 0))
    tape = tape.rotate(-6, expand=True, resample=Image.BICUBIC)
    sh.alpha_composite(tape, (sh.width // 2 - tape.width // 2, 2))
    return sh.rotate(angle, expand=True, resample=Image.BICUBIC)


# ----------------------------------------------------------------------------- doodles
def star(cx, cy, r):
    pts = []
    for i in range(9):
        a = i * math.pi / 4 - math.pi / 2
        rr = r if i % 2 == 0 else r * 0.28
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return np.array(pts)


def heart(cx, cy, s):
    t = np.linspace(0, 2 * math.pi, 80)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return np.c_[cx + x * s / 16, cy + y * s / 16]


def sweat(cx, cy, s):
    a = np.linspace(-0.15 * math.pi, 1.15 * math.pi, 30)
    arc = np.c_[cx + s * 0.55 * np.cos(a), cy + s * 0.55 * np.sin(a)]
    return np.vstack([arc, [[cx, cy - s * 1.2]]])


def arrow(c, p0, p1, bend, can, key, col=WHITE, w=9, p=1.0, outline=INK):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    mid = (p0 + p1) / 2 + np.array([-(p1 - p0)[1], (p1 - p0)[0]]) * bend
    pts = qbez(p0, mid, p1, 30)
    can.stroke(c, pts, w, col, p=min(1, p / 0.75), key=key + ("a",), outline=outline, sketch=False)
    if p > 0.75:
        d = p1 - mid
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        hp = ease((p - 0.75) / 0.25)
        for sgn in (1, -1):
            tip = p1 - d * 45 * hp + n * 28 * sgn * hp
            can.stroke(c, [p1, tip], w, col, key=key + ("h", sgn), outline=outline, sketch=False)


def burst(cx, cy, r0, r1, n, can, c, key, col=INK, w=6, p=1.0):
    for i in range(n):
        a = 2 * math.pi * i / n + 0.3
        can.stroke(c, [(cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
                       (cx + r1 * math.cos(a), cy + r1 * math.sin(a))], w, col, p=p, key=key + (i,), sketch=False)


def hand(cx, cy, side, s=1.0):
    """Mitten-style hand; side=-1 comes from left edge, +1 from right edge."""
    palm = ell(cx, cy, 100 * s, 66 * s, n=60)
    thumb = ell(cx - side * 20 * s, cy - 62 * s, 46 * s, 28 * s, n=40)
    edge = -40 if side < 0 else W + 40
    arm = np.array([[edge, cy - 55 * s], [cx + side * 40 * s, cy - 50 * s],
                    [cx + side * 40 * s, cy + 52 * s], [edge, cy + 60 * s]])
    return arm, thumb, palm


def draw_hand(can, c, cx, cy, side, key, s=1.0, p=1.0):
    arm, thumb, palm = hand(cx, cy, side, s)
    can.fill(c, arm, SKIN, key + ("arm",))
    can.stroke(c, arm[:2], 7, INK, p=p, key=key + ("a1",))
    can.stroke(c, arm[2:], 7, INK, p=p, key=key + ("a2",))
    can.shape(c, thumb, SKIN, key + ("th",), p=p)
    can.shape(c, palm, SKIN, key + ("pa",), p=p)


def cup_body(cx, top, h, wt, wb):
    r = 40
    left = np.array([[cx - wt, top], [cx - wb, top + h - r]])
    bl = qbez((cx - wb, top + h - r), (cx - wb + 4, top + h), (cx - wb + r, top + h), 12)
    br = qbez((cx + wb - r, top + h), (cx + wb - 4, top + h), (cx + wb, top + h - r), 12)
    right = np.array([[cx + wb, top + h - r], [cx + wt, top]])
    rim = ell(cx, top, wt, wt * 0.2, 0, math.pi, 30)[::-1]  # front half of rim, right->left
    return np.vstack([left, bl, br, right, rim])


def draw_cup(can, c, cx, top, h, wt, wb, key, p=1.0, col=CLAY, handle=True, face=None, flower=False):
    body = cup_body(cx, top, h, wt, wb)
    if handle:
        hpts = cbez((cx + wt - 12, top + h * 0.18), (cx + wt + 140, top + h * 0.1),
                    (cx + wt + 120, top + h * 0.75), (cx + (wt + wb) / 2 - 4, top + h * 0.7), 30)
        can.stroke(c, hpts, 26, col, p=p, key=key + ("hf",), sketch=False, amp=1.5)
        can.stroke(c, hpts, 7, INK, p=p, key=key + ("h",))
    can.shape(c, body, col, key + ("b",), p=p)
    rim = ell(cx, top, wt, wt * 0.2)
    if p > 0.6:
        can.fill(c, rim, (0.77, 0.70, 0.61), key + ("rf",), alpha=ease((p - 0.6) / 0.4))
    can.stroke(c, rim, 7, INK, p=p, key=key + ("r",))
    if flower and p >= 1:
        fx, fy = cx - 10, top + h * 0.5
        for i in range(5):
            a = i * 2 * math.pi / 5
            can.shape(c, ell(fx + 34 * math.cos(a), fy + 34 * math.sin(a), 26, 26, n=24), (0.62, 0.60, 0.58),
                      key + ("pet", i), w=4)
        can.fill(c, ell(fx, fy, 12, 12, n=20), (0.42, 0.33, 0.28), key + ("fc",))
        can.stroke(c, qbez((fx, fy + 40), (fx + 6, fy + 100), (fx - 4, top + h - 20)), 6, (0.45, 0.32, 0.25), key=key + ("st",))
        can.stroke(c, qbez((fx + 2, fy + 110), (fx + 45, fy + 80), (fx + 60, fy + 110)), 6, (0.45, 0.32, 0.25), key=key + ("lf",))
    if face and p >= 1:
        ey = top + h * 0.42
        if face == "happy":
            for sx in (-1, 1):
                can.fill(c, ell(cx + sx * 55, ey, 11, 14, n=20), INK, key + ("e", sx), amp=1)
            can.stroke(c, qbez((cx - 30, ey + 50), (cx, ey + 80), (cx + 30, ey + 50)), 6, INK, key=key + ("m",))
            for sx in (-1, 1):
                can.fill(c, ell(cx + sx * 95, ey + 40, 22, 12, n=20), PINK, key + ("ch", sx), alpha=0.6)
        else:
            for sx in (-1, 1):
                can.stroke(c, [(cx + sx * 55 - 14, ey - 12), (cx + sx * 55 + 14, ey + 12)], 6, INK, key=key + ("x1", sx))
                can.stroke(c, [(cx + sx * 55 - 14, ey + 12), (cx + sx * 55 + 14, ey - 12)], 6, INK, key=key + ("x2", sx))
            can.stroke(c, qbez((cx - 32, ey + 75), (cx, ey + 45), (cx + 32, ey + 75)), 6, INK, key=key + ("m2",))


def subtitle(can, s, y=1470, alpha=1.0):
    can.text(s, 530, y, size=70, font=GAEGU, col=(255, 255, 255), stroke=8, stroke_col=(52, 38, 28),
             jitter=0, alpha=alpha)


# ----------------------------------------------------------------------------- scenes
class Scene:
    cap_y = 1470

    def __init__(self, vo, dur, vo_at=0.35, captions=None):
        self.vo, self.dur, self.vo_at = vo, dur, vo_at
        self.captions = captions or []

    def draw(self, t, f):
        raise NotImplementedError


# 1 ---------------------------------------------------------------- hook (drawn)
class Hook(Scene):
    def draw(self, t, f):
        can = Canvas(paper(f), f)
        can.text("도자기 컵 하나", 540, 330, size=130, p=prog(t, 0.1, 0.8), key=1)
        can.text("만드는 데 얼마나 걸릴까?", 540, 470, size=104, p=prog(t, 0.6, 1.5), key=2, col=(170, 70, 50))
        with can.vec() as c:
            can.stroke(c, qbez((260, 540), (540, 560), (840, 535)), 8, (0.67, 0.27, 0.2), p=prog(t, 1.4, 1.8), key=("ul",))
            draw_cup(can, c, 520, 820, 380, 220, 165, ("cup",), p=prog(t, 0.2, 1.8))
            # clock with spinning hands (time passing)
            cp = prog(t, 1.6, 2.1)
            if cp > 0:
                can.shape(c, ell(215, 1250, 85, 85), (0.99, 0.97, 0.92), ("clk",), p=cp)
                a1 = t * 9
                a2 = t * 1.2
                can.stroke(c, [(215, 1250), (215 + 60 * math.sin(a1), 1250 - 60 * math.cos(a1))], 7, INK, p=cp, key=("m",), sketch=False)
                can.stroke(c, [(215, 1250), (215 + 38 * math.sin(a2), 1250 - 38 * math.cos(a2))], 9, RED, p=cp, key=("h",), sketch=False)
                for i in range(4):
                    a = i * math.pi / 2
                    can.stroke(c, [(215 + 68 * math.sin(a), 1250 - 68 * math.cos(a)), (215 + 78 * math.sin(a), 1250 - 78 * math.cos(a))], 5, INK, key=("tk", i), sketch=False)
        for i, (x, y, t0, r) in enumerate([(220, 860, 1.5, -12), (880, 760, 1.8, 10), (880, 1230, 2.1, 6)]):
            sc = ease_out_back(prog(t, t0, t0 + 0.3))
            can.text("?", x, y, size=190, scale=sc, rot=r + 4 * math.sin(t * 5 + i), key=10 + i, col=(170, 70, 50))
        return can


# 2 ---------------------------------------------------------------- clay block (photo)
class ClayPhoto(Scene):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ph = Photo("clay_block.jpg", 1.0, 1.08, (0.5, 0.42), (0.47, 0.38))

    def draw(self, t, f):
        u = t / self.dur
        col = self.ph.frame(u)
        rv = ease(prog(t, 0.35, 1.0))
        base = Image.blend(self.ph.frame(u, self.ph.sketch()), col, rv) if rv < 1 else col
        can = Canvas(base, f)
        tl = self.ph.pt(u, 0.27, 0.07)
        br = self.ph.pt(u, 0.64, 0.53)
        with can.vec() as c:
            cx, cy = (tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2
            loop = ell(cx, cy, (br[0] - tl[0]) / 2 + 70, (br[1] - tl[1]) / 2 + 80, -1.7, -1.7 + 2 * math.pi * 1.08, 90)
            can.stroke(c, loop, 9, YELLOW, p=prog(t, 0.9, 1.6), key=("lp",), outline=INK, sketch=False)
            arrow(c, (900, 1270), (cx + 120, br[1] + 70), -0.25, can, ("ar",), col=WHITE, p=prog(t, 1.3, 1.8))
        can.text("시작은 흙 한 덩어리", 640, 1290, size=104, col=(255, 255, 255), stroke=6, stroke_col=(52, 38, 28),
                 p=prog(t, 1.4, 2.2), key=3)
        return can


# 3 ---------------------------------------------------------------- centering (drawn)
class Centering(Scene):
    def draw(self, t, f):
        D = self.dur
        can = Canvas(paper(f), f)
        tc = 0.74 * D                                   # collapse moment
        shake = (0.0, 0.0)
        if tc < t < tc + 0.3:
            shake = tuple(rng("sh", f).normal(0, 10, 2))
        can.text("STEP 1", 540, 255, size=72, p=prog(t, 0.0, 0.3), key=1, col=(150, 120, 100))
        can.text("물레 위 중심 잡기", 540, 360, size=128, p=prog(t, 0.1, 0.9), key=2)
        with can.vec() as c:
            c.translate(*shake)
            can.stroke(c, qbez((250, 430), (540, 455), (830, 425)), 8, (0.67, 0.27, 0.2), p=prog(t, 0.8, 1.2), key=("ul",))
            # wheel: splash pan + head
            can.shape(c, np.vstack([ell(540, 1200, 430, 120, 0, math.pi, 40), [[110, 1200]]]), (0.95, 0.95, 0.94), ("pan",))
            can.shape(c, np.vstack([[[150, 1200], [930, 1200]], ell(540, 1200, 390, 105, 0, math.pi, 40)]), (0.91, 0.91, 0.90), ("pan2",))
            can.shape(c, ell(540, 1165, 300, 72), GREY, ("head",))
            # speed lines on the wheel head
            for i in range(3):
                a0 = t * 7 + i * 2.1
                can.stroke(c, ell(540, 1165, 250 - i * 40, 58 - i * 9, a0, a0 + 1.0, 20), 5, (0.45, 0.47, 0.5), key=("sp", i), sketch=False)
            # clay lump
            if t < tc:
                A = 12 + 55 * ease(t / tc)
                dx = A * math.sin(2 * math.pi * 2.2 * t)
                hgt, lean = 330, 0
            else:
                k = ease((t - tc) / 0.35)
                dx = 0 + 150 * k
                hgt = 330 - 170 * k
                lean = k
            base_y = 1165
            prof_l, prof_r = [], []
            for sidx in np.linspace(0, 1, 18):
                wv = 165 * (1 - 0.32 * sidx ** 2) * (1 + 0.45 * lean * sidx)
                x = 540 + dx * sidx ** 1.4
                y = base_y - hgt * sidx
                prof_l.append((x - wv, y))
                prof_r.append((x + wv, y))
            topc = (540 + dx, base_y - hgt)
            wt = 165 * (1 - 0.32) * (1 + 0.45 * lean)
            dome = ell(topc[0], topc[1], wt, 40 + 30 * lean, math.pi, 2 * math.pi, 24)
            lump = np.vstack([prof_l, dome, prof_r[::-1]])
            can.shape(c, lump, CLAY, ("lump",))
            # throwing rings on clay
            for i, s in enumerate((0.3, 0.55, 0.78)):
                y = base_y - hgt * s
                x = 540 + dx * s ** 1.4
                wv = 165 * (1 - 0.32 * s ** 2) * (1 + 0.45 * lean * s)
                ph = (t * 3 + i * 0.3) % 1
                can.stroke(c, qbez((x - wv * 0.8 + ph * 40, y), (x, y + 14), (x + wv * 0.6 + ph * 40, y)), 4, CLAY_D, key=("rg", i), sketch=False)
            # hands pressing from both sides (follow the lump until collapse)
            hy = base_y - hgt * 0.45
            hx = 540 + dx * 0.45 ** 1.4
            wv = 165 * (1 - 0.32 * 0.2) * (1 + 0.45 * lean * 0.45)
            off = 0 if t < tc else 120 * ease((t - tc) / 0.3)   # hands jump back
            draw_hand(can, c, hx - wv - 70 - off, hy, -1, ("hl",))
            draw_hand(can, c, hx + wv + 70 + off, hy, +1, ("hr",))
            # sweat drops
            for i, (sx, sy, t0) in enumerate([(300, 720, 1.2), (790, 690, 2.0), (250, 640, 3.0), (840, 790, 3.6)]):
                if t > t0:
                    fall = ((t - t0) * 0.9) % 1.0
                    can.shape(c, sweat(sx, sy + fall * 120, 30), BLUE, ("sw", i), w=5, fill_alpha=0.85)
            # wobble lines
            if t < tc:
                for sgn in (-1, 1):
                    xx = 540 + sgn * 300 + dx * 0.6
                    can.stroke(c, qbez((xx, 840), (xx + sgn * 25, 905), (xx, 970)), 6, INK, key=("wl", sgn), sketch=False)
                    can.stroke(c, qbez((xx + sgn * 30, 860), (xx + sgn * 50, 905), (xx + sgn * 30, 950)), 5, INK, key=("wl2", sgn), sketch=False)
        if 1.0 < t < tc:
            can.text("흔들흔들~", 790, 640 + 10 * math.sin(t * 6), size=100, rot=-8, key=5, col=(170, 70, 50),
                     scale=ease_out_back(prog(t, 1.0, 1.3)))
        if 3.2 < t < tc:
            can.text("으아아!", 260, 560, size=96, rot=10, key=6, scale=ease_out_back(prog(t, 3.2, 3.5)))
        if t >= tc + 0.1:
            can.text("털썩...", 760, 880, size=120, rot=-6, key=7, scale=ease_out_back(prog(t, tc + 0.1, tc + 0.4)))
        return can


# 4 ---------------------------------------------------------------- drying (photos)
class Drying(Scene):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.a = Photo("cups_drying.jpg", 1.0, 1.08, (0.45, 0.5), (0.4, 0.52))
        self.b = Photo("plates.jpg", 1.08, 1.0, (0.45, 0.6), (0.5, 0.55))
        self.split = self.dur * 0.5

    def draw(self, t, f):
        if t < self.split:
            u = t / self.split
            can = Canvas(self.a.frame(u), f)
            can.text("말리는 중...", 540, 300, size=120, col=(255, 255, 255), stroke=6, stroke_col=(52, 38, 28),
                     p=prog(t, 0.1, 0.7), key=1)
            for i, (x, y) in enumerate([(250, 850), (770, 980), (420, 1220)]):
                tt = t - 0.3 - i * 0.25
                if tt > 0:
                    can.text("z", x + 40 * tt, y - 70 * tt, size=70, col=(255, 255, 255), stroke=5, stroke_col=(52, 38, 28), key=10 + i)
                    can.text("Z", x + 30 + 60 * tt, y - 70 - 90 * tt, size=95, col=(255, 255, 255), stroke=5, stroke_col=(52, 38, 28), key=20 + i)
        else:
            tt = t - self.split
            u = tt / (self.dur - self.split)
            can = Canvas(self.b.frame(u), f)
            can.text("기다리고...", 330, 300, size=110, col=(255, 255, 255), stroke=6, stroke_col=(52, 38, 28),
                     p=prog(tt, 0.0, 0.5), key=2, rot=4)
            can.text("또 기다리고...", 680, 430, size=110, col=(255, 255, 255), stroke=6, stroke_col=(52, 38, 28),
                     p=prog(tt, 0.6, 1.2), key=3, rot=-3)
        return can


# 5 ---------------------------------------------------------------- trimming (video + photo)
class Trimming(Scene):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.v = Video("trimming.mov", 0.2)
        self.ph = Photo("trim_shavings.jpg", 1.0, 1.06, (0.5, 0.45), (0.5, 0.42))
        self.split = 2.3

    def draw(self, t, f):
        if t < self.split:
            fr = self.v.next()
            can = Canvas(fr, f)
            with can.vec() as c:
                for i in range(2):
                    a0 = -t * 4 + i * math.pi
                    can.stroke(c, ell(540, 1080, 390, 110, a0, a0 + 1.6, 30), 9, YELLOW, key=("rot", i), outline=INK, sketch=False)
            can.text("뒤집어서 굽 깎기", 540, 330, size=115, col=(255, 255, 255), stroke=6, stroke_col=(52, 38, 28),
                     p=prog(t, 0.1, 0.8), key=1)
        else:
            tt = t - self.split
            u = tt / (self.dur - self.split)
            can = Canvas(self.ph.frame(u), f)
            with can.vec() as c:
                p0 = self.ph.pt(u, 0.5, 0.06)
                loop = ell(p0[0], p0[1] + 30, 500, 175, 0.3, 0.3 + 2 * math.pi * 1.05, 90)
                can.stroke(c, loop, 10, YELLOW, p=prog(tt, 0.1, 0.7), key=("lp",), outline=INK, sketch=False)
                arrow(c, (300, 640), (380, 330), 0.25, can, ("ar",), p=prog(tt, 0.5, 0.9))
            can.text("이만큼!!", 300, 720, size=130, col=(255, 255, 255), stroke=6, stroke_col=(52, 38, 28),
                     p=1, scale=ease_out_back(prog(tt, 0.7, 1.0)), rot=6, key=2)
        return can


# 6 ---------------------------------------------------------------- vase sponge (video)
class VaseVideo(Scene):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.v = Video("vase_sponge.mp4", 0.4)

    def draw(self, t, f):
        can = Canvas(self.v.next(), f)
        # tally marks (Korean 正 count) appearing over time
        n_strokes = int(clamp(t / (self.dur - 0.6)) * 20)
        with can.vec() as c:
            for i in range(n_strokes):
                g, s = divmod(i, 5)
                ox, oy = 110 + g * 130, 260
                segs = [[(ox, oy), (ox + 90, oy)], [(ox + 45, oy), (ox + 45, oy + 110)],
                        [(ox + 45, oy + 52), (ox + 85, oy + 52)], [(ox + 10, oy + 40), (ox + 10, oy + 110)],
                        [(ox - 5, oy + 110), (ox + 100, oy + 110)]][s]
                can.stroke(c, segs, 10, WHITE, key=("ty", i), outline=INK, sketch=False)
        can.text("손이 가는 횟수", 330, 170, size=78, col=(255, 255, 255), stroke=5, stroke_col=(52, 38, 28),
                 p=prog(t, 0.0, 0.5), key=1)
        if t > self.dur - 1.4:
            can.text("...셀 수 없음", 780, 470, size=86, col=(255, 230, 120), stroke=5, stroke_col=(52, 38, 28),
                     scale=ease_out_back(prog(t, self.dur - 1.4, self.dur - 1.1)), rot=-5, key=2)
        return can


# 7 ---------------------------------------------------------------- kiln (drawn)
class Kiln(Scene):
    def draw(self, t, f):
        D = self.dur
        can = Canvas(paper(f), f)
        vo = lambda frac: self.vo_at + frac * self.vo_len
        steps = [("초벌", 230, vo(0.12)), ("유약", 540, vo(0.24)), ("재벌", 850, vo(0.36))]
        with can.vec() as c:
            for i in range(2):
                tt = steps[i][2]
                p = prog(t, tt + 0.15, tt + 0.45)
                can.stroke(c, [(steps[i][1] + 85, 340), (steps[i + 1][1] - 85, 340)], 7, INK, p=p, key=("st", i), sketch=False)
                if p >= 1:
                    x = steps[i + 1][1] - 85
                    can.stroke(c, [(x, 330), (x - 26, 312)], 7, INK, key=("sh1", i), sketch=False)
                    can.stroke(c, [(x, 330), (x - 26, 348)], 7, INK, key=("sh2", i), sketch=False)
            # kiln body
            kp = prog(t, 0.0, 0.8)
            can.shape(c, np.array([[260, 600], [820, 600], [820, 1260], [260, 1260]]), (0.66, 0.68, 0.72), ("kb",), p=kp)
            can.shape(c, np.array([[330, 670], [750, 670], [750, 1110], [330, 1110]]), (0.24, 0.18, 0.17), ("kw",), p=kp)
            for x in (300, 780):
                can.stroke(c, [(x, 1260), (x, 1330)], 12, INK, p=kp, key=("leg", x))
            can.shape(c, np.array([[640, 600], [710, 600], [710, 520], [640, 520]]), (0.6, 0.62, 0.66), ("ch",), p=kp)
            # smoke
            for i in range(3):
                ph = (t * 0.6 + i / 3) % 1
                can.stroke(c, ell(675 + 30 * math.sin(ph * 6 + i), 500 - ph * 160, 22 + ph * 20, 16 + ph * 12, n=30), 5,
                           (0.55, 0.55, 0.55), alpha=1 - ph, key=("sm", i), sketch=False)
            # flames (flicker on boil steps)
            if kp >= 1:
                fl = prog(t, 0.8, 1.6)
                for i, x in enumerate((400, 540, 680)):
                    r = rng("fl", i, can.b)
                    h = (150 + 90 * r.random()) * fl
                    if h > 5:
                        outer = np.vstack([qbez((x - 55, 1100), (x - 70, 1100 - h * 0.6), (x + r.normal(0, 15), 1100 - h), 16),
                                           qbez((x + r.normal(0, 15), 1100 - h), (x + 70, 1100 - h * 0.6), (x + 55, 1100), 16)])
                        can.fill(c, outer, ORANGE, ("fo", i), alpha=0.95)
                        inner = np.vstack([qbez((x - 28, 1100), (x - 34, 1100 - h * 0.35), (x, 1100 - h * 0.55), 12),
                                           qbez((x, 1100 - h * 0.55), (x + 34, 1100 - h * 0.35), (x + 28, 1100), 12)])
                        can.fill(c, inner, YELLOW, ("fi", i), alpha=0.95)
                # cup silhouette on shelf
                can.stroke(c, [(345, 980), (735, 980)], 8, (0.5, 0.45, 0.42), key=("shelf",), sketch=False)
                can.shape(c, cup_body(540, 870, 105, 70, 52), CLAY, ("kc",), w=5)
            # thermometer
            tp = prog(t, 0.6, 1.0)
            can.shape(c, np.vstack([ell(930, 640, 26, 26, math.pi, 2 * math.pi, 20), [[956, 1180]], ell(930, 1210, 46, 46, -0.95, math.pi + 0.95, 30), [[904, 1180]]]),
                      (0.99, 0.98, 0.96), ("th",), p=tp, w=6)
            lvl = 1180 - 520 * ease(prog(t, 1.2, D - 0.6))
            if tp >= 1:
                can.fill(c, ell(930, 1210, 32, 32, n=30), RED, ("bulb",), amp=1)
                can.fill(c, np.array([[918, 1185], [942, 1185], [942, lvl], [918, lvl]]), RED, ("lv",), amp=1)
        for name, x, tt in steps:
            can.text(name, x, 330, size=110, key=hash(name) % 100, scale=ease_out_back(prog(t, tt, tt + 0.3)),
                     col=(170, 70, 50) if name == "유약" else (56, 41, 31))
        if t > 1.4:
            can.text("1250도", 900, 560, size=74, key=9, scale=ease_out_back(prog(t, D - 1.2, D - 0.9)))
        can.text("가마에 두 번!", 540, 190, size=96, key=8, scale=ease_out_back(prog(t, vo(0.62), vo(0.62) + 0.3)), rot=-3)
        return can


# 8 ---------------------------------------------------------------- glaze tests (photo)
class Glaze(Scene):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ph = Photo("glaze_tests.jpg", 1.0, 1.07, (0.5, 0.55), (0.5, 0.6))

    def draw(self, t, f):
        u = t / self.dur
        can = Canvas(self.ph.frame(u), f)
        xs = [0.27, 0.485, 0.71]
        labels = ["1번?", "2번?", "3번?"]
        with can.vec() as c:
            for i, nx in enumerate(xs):
                p = self.ph.pt(u, nx, 0.525)
                t0 = 0.4 + i * 0.55
                if t > t0:
                    arrow(c, (p[0], p[1] - 150), (p[0], p[1] - 30), 0.0, can, ("ga", i), p=prog(t, t0, t0 + 0.35))
            if t > 2.4:
                p = self.ph.pt(u, 0.27, 0.66)
                can.stroke(c, heart(p[0] + 150, p[1] - 250, 45), 8, PINK, key=("ht",), outline=INK, sketch=False,
                           p=prog(t, 2.4, 2.8), closed=True)
        for i, nx in enumerate(xs):
            p = self.ph.pt(u, nx, 0.525)
            t0 = 0.4 + i * 0.55
            can.text(labels[i], p[0], p[1] - 210, size=96, col=(255, 255, 255), stroke=6, stroke_col=(52, 38, 28),
                     scale=ease_out_back(prog(t, t0, t0 + 0.3)), key=i, rot=(-1) ** i * 5)
        can.text("유약 테스트", 540, 300, size=120, col=(255, 255, 255), stroke=6, stroke_col=(52, 38, 28),
                 p=prog(t, 0.0, 0.6), key=5)
        return can


# 9 ---------------------------------------------------------------- crack (drawn)
class Crack(Scene):
    def draw(self, t, f):
        D = self.dur
        tc = self.vo_at + self.vo_len * 0.62
        can = Canvas(paper(f), f)
        dx = dy = 0
        if tc < t < tc + 0.35:
            r = rng("sh", f)
            dx, dy = r.normal(0, 14, 2)
        # calendar flipping to day 30
        day = int(1 + 29 * ease(prog(t, 0.1, 1.4)))
        with can.vec() as c:
            c.translate(dx, dy)
            can.shape(c, np.array([[150, 230], [430, 230], [430, 510], [150, 510]]), (0.99, 0.98, 0.95), ("cal",))
            can.fill(c, np.array([[150, 230], [430, 230], [430, 300], [150, 300]]), RED, ("calh",))
            for x in (210, 370):
                can.stroke(c, [(x, 205), (x, 255)], 9, INK, key=("ring", x), sketch=False)
            # door opening
            op = ease(prog(t, 1.2, 1.9))
            if op < 1:
                dw = 560 * (1 - op)
                can.shape(c, np.array([[260, 620], [260 + dw, 640], [260 + dw, 1300], [260, 1320]]), (0.6, 0.62, 0.66), ("door",))
            face = "sad" if t > tc else "happy"
            if op > 0.3:
                draw_cup(can, c, 520, 820, 380, 220, 165, ("ccup",), face=face)
                # crack line drawn quickly
                cp = prog(t, tc - 0.05, tc + 0.15)
                if cp > 0:
                    crack = np.array([[440, 830], [470, 900], [430, 950], [490, 1020], [455, 1080], [510, 1140]])
                    can.stroke(c, crack, 9, INK, p=cp, key=("cr",), amp=0.6)
                    can.stroke(c, np.array([[490, 1020], [560, 1050], [590, 1110]]), 6, INK, p=cp, key=("cr2",), amp=0.6)
                if t < tc and op >= 1:
                    for i, (x, y) in enumerate([(240, 760), (860, 900), (300, 1180)]):
                        sc = 0.6 + 0.4 * abs(math.sin(t * 4 + i))
                        can.shape(c, star(x, y, 40 * sc), YELLOW, ("sp", i), w=5)
            if t > tc:
                burst(820, 700, 120, 190, 9, can, c, ("bs",), col=RED, p=prog(t, tc, tc + 0.2))
                # tears
                for sx in (-1, 1):
                    ph = ((t - tc) * 1.3) % 1
                    can.shape(c, sweat(520 + sx * 55, 1010 + ph * 90, 20), BLUE, ("tear", sx), w=4)
        can.text("D+%d" % day, 290, 410, size=110, key=1, col=(56, 41, 31))
        if 1.9 < t < tc:
            can.text("두근두근", 540, 680, size=110, key=2, scale=1 + 0.06 * math.sin(t * 14), col=(170, 70, 50))
        if t > tc:
            can.text("쩍!", 820, 700, size=220, key=3, col=(200, 60, 50), scale=ease_out_back(prog(t, tc, tc + 0.25)), rot=-10)
        return can


# 10 --------------------------------------------------------------- finished cup (video)
class Proud(Scene):
    cap_y = 470

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.v = Video("flower_cup.mov", 0.3)

    def draw(self, t, f):
        can = Canvas(self.v.next(), f)
        with can.vec() as c:
            for i, (x, y, t0, r) in enumerate([(230, 1000, 0.6, 45), (860, 1080, 0.9, 55), (280, 1560, 1.2, 38),
                                                (830, 1560, 1.5, 42), (540, 900, 1.8, 34)]):
                if t > t0:
                    sc = ease_out_back(prog(t, t0, t0 + 0.3)) * (0.8 + 0.2 * math.sin(t * 5 + i))
                    can.stroke(c, star(x, y, r * sc), 7, YELLOW, key=("st", i), outline=INK, sketch=False, closed=True)
        can.text("완성!!", 540, 780, size=170, col=(255, 230, 120), stroke=7, stroke_col=(52, 38, 28),
                 scale=ease_out_back(prog(t, 0.3, 0.65)), rot=-4, key=1)
        return can


# 11 --------------------------------------------------------------- vase polaroid
class Proud2(Scene):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.im = load("vase_finished.jpg")
        self.pol = polaroid(self.im, 780, 3)
        self.pol_sk = polaroid(sketchify(self.im), 780, 3)

    def draw(self, t, f):
        can = Canvas(paper(f), f)
        rv = ease(prog(t, 0.2, 0.9))
        y = 220 + 25 * (1 - ease(prog(t, 0, 0.5)))
        pol = Image.blend(self.pol_sk, self.pol, rv) if rv < 1 else self.pol
        _paste(can.img, pol, 540 - pol.width // 2, int(y))
        can.text("내가 만든 화병", 545, int(y) + pol.height - 150, size=84, key=1, p=prog(t, 0.6, 1.3), rot=3)
        with can.vec() as c:
            for i, (x, y2, t0, s) in enumerate([(130, 400, 0.9, 40), (950, 520, 1.1, 50), (120, 1180, 1.3, 36), (960, 1240, 1.5, 44)]):
                if t > t0:
                    yy = y2 - 30 * (t - t0)
                    can.shape(c, heart(x, yy, s * ease_out_back(prog(t, t0, t0 + 0.3))), PINK, ("h", i), w=6)
        return can


# 12 --------------------------------------------------------------- outro (drawn)
class Outro(Scene):
    def draw(self, t, f):
        can = Canvas(paper(f), f)
        with can.vec() as c:
            draw_cup(can, c, 540, 860, 360, 210, 160, ("ocup",), p=prog(t, 0.0, 0.7), flower=True, handle=False)
            if t > 0.5:
                draw_hand(can, c, 300, 1110, -1, ("ohl",), p=prog(t, 0.5, 0.9))
                draw_hand(can, c, 780, 1110, +1, ("ohr",), p=prog(t, 0.5, 0.9))
            # steam that turns into a heart
            hp = prog(t, 0.8, 1.8)
            for i, x in enumerate((470, 610)):
                ph = t * 1.5 + i
                pts = np.array([(x + 18 * math.sin(ph + k * 0.5), 820 - k * 14) for k in range(12)])
                can.stroke(c, pts, 6, (0.6, 0.55, 0.5), key=("stm", i), alpha=0.8 * (1 - hp), sketch=False)
            if hp > 0:
                can.shape(c, heart(540, 690, 85), PINK, ("bigh",), p=hp)
            can.stroke(c, qbez((210, 520), (540, 545), (870, 515)), 9, (0.67, 0.27, 0.2), p=prog(t, 1.2, 1.6), key=("ul",))
        can.text("내 취미는", 540, 300, size=110, p=prog(t, 0.2, 0.7), key=1)
        can.text("도자기 만들기!", 540, 440, size=160, p=prog(t, 0.6, 1.3), key=2, col=(170, 70, 50))
        return can


# ----------------------------------------------------------------------------- timeline
def build(vo):
    d = vo["durations"]
    pad = 0.55
    defs = [
        (Hook, "hook", 0.5, 4.2, [("도자기 컵 하나 만드는 데,\n얼마나 걸릴 것 같아요?", 1.0)]),
        (ClayPhoto, "clay", 0.35, None, [("시작은 이렇게,\n투박한 흙 한 덩어리예요.", 1.0)]),
        (Centering, "center", 0.4, None, [("물레 위에서 중심 잡기부터가\n첫 번째 고비.", 0.45),
                                          ("힘을 조금만 놓쳐도,\n흙이 비틀비틀 무너져요.", 0.55)]),
        (Drying, "dry", 0.3, None, [("겨우 모양을 만들면,\n꾸덕하게 마를 때까지", 0.55), ("또 기다려야 하고요.", 0.45)]),
        (Trimming, "trim", 0.3, None, [("다 마르면\n뒤집어서 굽을 깎아요.", 0.55), ("깎아낸 흙이\n이만큼이에요.", 0.45)]),
        (VaseVideo, "vase", 0.3, None, [("손잡이를 붙이고,\n스펀지로 다듬고.", 0.6), ("손이 몇십 번은 가요.", 0.4)]),
        (Kiln, "kiln", 0.35, None, [("그다음엔 초벌, 유약, 재벌.", 0.45), ("가마에만 두 번\n들어갔다 나와야 해요.", 0.55)]),
        (Glaze, "glaze", 0.3, None, [("원하는 색 하나 찾으려고,\n유약 테스트도 몇 번씩 하고요.", 1.0)]),
        (Crack, "crack", 0.5, None, [("그렇게 한 달을 기다렸는데,", 0.45), ("가마를 열면\n금이 가 있을 때도 있어요.", 0.55)]),
        (Proud, "proud", 0.4, None, [("그래도, 내 손으로\n끝까지 완성한 걸", 0.5), ("처음 손에 쥐는 순간.", 0.5)]),
        (Proud2, "proud2", 0.3, None, [("그 뿌듯함 때문에,\n저는 또 물레 앞에 앉아요.", 1.0)]),
        (Outro, "outro", 0.6, 3.6, [("제 취미는,\n도자기 만들기예요.", 1.0)]),
    ]
    scenes = []
    for cls, key, at, dmin, caps in defs:
        dur = max(dmin or 0, at + d[key] + pad)
        if key == "crack":
            dur += 0.6
        sc = cls(key, dur, at, caps)
        sc.vo_len = d[key]
        scenes.append(sc)
    # Kiln / Crack read vo_len inside draw; Drying uses split from dur set in __init__ (dur known already)
    return scenes


def render(vo_dir, out, only=None, stills=None):
    vo = json.load(open(os.path.join(vo_dir, "narration.json")))
    scenes = build(vo)
    t0 = 0
    timeline = []
    for s in scenes:
        timeline.append((t0, s))
        t0 += s.dur
    total = t0
    print("total duration %.2fs" % total)
    if stills:
        for idx, tt in stills:
            st, s = timeline[idx]
            f = int((st + tt) * FPS)
            can = s.draw(tt, f)
            add_captions(can, s, tt)
            can.img.convert("RGB").save(os.path.join(stills_dir(out), f"s{idx:02d}_{tt:.1f}.jpg"), quality=85)
        return
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    gf = 0
    for i, (st, s) in enumerate(timeline):
        n = int(round((st + s.dur) * FPS)) - int(round(st * FPS))
        for k in range(n):
            tt = k / FPS
            if only is not None and i != only:
                gf += 1
                continue
            can = s.draw(tt, gf)
            add_captions(can, s, tt)
            img = can.img.convert("RGB")
            # quick punch-in on cuts
            if k < 4 and i > 0:
                z = 1 + 0.035 * (1 - k / 4)
                img = img.resize((W, H), Image.BICUBIC, box=((W - W / z) / 2, (H - H / z) / 2, (W + W / z) / 2, (H + H / z) / 2))
            enc.stdin.write(img.tobytes())
            gf += 1
        print(f"scene {i} {s.vo} done", flush=True)
    enc.stdin.close()
    enc.wait()
    json.dump([{"start": st, "dur": s.dur, "vo": s.vo, "vo_at": s.vo_at, "vo_len": s.vo_len, "type": type(s).__name__}
               for st, s in timeline], open(out + ".timeline.json", "w"), indent=1)


def stills_dir(out):
    d = os.path.join(os.path.dirname(out) or ".", "stills")
    os.makedirs(d, exist_ok=True)
    return d


def add_captions(can, s, t):
    ts = s.vo_at
    for text, frac in s.captions:
        te = ts + frac * s.vo_len
        last = text == s.captions[-1][0]
        if ts - 0.05 <= t < (te if not last else s.dur - 0.1):
            a = ease(prog(t, ts - 0.05, ts + 0.12))
            subtitle(can, text, y=s.cap_y, alpha=a)
        ts = te


if __name__ == "__main__":
    vo_dir, out = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "--stills":
        pairs = [tuple(map(float, x.split(":"))) for x in sys.argv[4:]]
        render(vo_dir, out, stills=[(int(a), b) for a, b in pairs])
    else:
        render(vo_dir, out)
