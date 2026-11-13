"""Mix narration + synthesized lo-fi BGM + hand-drawn SFX, then mux with the rendered video.

usage: python3 mix.py <narration_dir> <video_silent.mp4> <out.mp4>
"""
import json, os, subprocess, sys
import numpy as np
import soundfile as sf

vo_dir, vid, out = sys.argv[1:4]
SR = 44100
tl = json.load(open(vid + ".timeline.json"))
total = tl[-1]["start"] + tl[-1]["dur"]
N = int(total * SR) + SR
rng = np.random.default_rng(7)


def at(t):
    return int(t * SR)


def bandnoise(dur, lo, hi):
    n = int(dur * SR)
    x = rng.normal(0, 1, n)
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1 / SR)
    X[(fr < lo) | (fr > hi)] = 0
    y = np.fft.irfft(X, n)
    return y / (np.abs(y).max() + 1e-9)


def add(buf, x, t, g=1.0):
    i = at(t)
    j = min(len(buf), i + len(x))
    if j > i:
        buf[i:j] += x[: j - i] * g


# ------------------------------------------------------------------ narration
voice = np.zeros(N)
vo_spans = []
for s in tl:
    x, sr = sf.read(os.path.join(vo_dir, s["vo"] + ".wav"))
    assert sr == SR
    t0 = s["start"] + s["vo_at"]
    add(voice, x, t0, 1.0)
    vo_spans.append((t0, t0 + len(x) / SR))

# ------------------------------------------------------------------ BGM (soft plucked arpeggios + pad)
bgm = np.zeros(N)
bpm = 84
beat = 60 / bpm
mid = lambda m: 440 * 2 ** ((m - 69) / 12)
chords = [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60], [48, 52, 55, 59]]  # Fmaj7 Em7 Dm7 Cmaj7
pattern = [0, 2, 1, 3, 2, 1, 3, 2]


def pluck(f, dur=1.6, vel=1.0):
    t = np.arange(int(dur * SR)) / SR
    env = np.exp(-t * 3.2) * np.minimum(1, t / 0.004)
    y = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) * np.exp(-t * 6)
         + 0.12 * np.sin(6 * np.pi * f * t) * np.exp(-t * 9))
    return y * env * vel


bar = 0
t = 0.0
while t < total:
    ch = chords[bar % 4]
    for k, idx in enumerate(pattern):
        note = ch[idx] + 12
        add(bgm, pluck(mid(note), vel=0.9 if k % 2 == 0 else 0.6), t + k * beat / 2, 0.22)
    # bass + pad
    tb = np.arange(int(4 * beat * SR)) / SR
    penv = np.minimum(1, tb / 0.6) * np.minimum(1, (4 * beat - tb) / 0.6)
    pad = sum(np.sin(2 * np.pi * mid(n) * tb) for n in ch) / 4
    add(bgm, pad * penv, t, 0.10)
    add(bgm, pluck(mid(ch[0] - 12), dur=4 * beat) , t, 0.35)
    t += 4 * beat
    bar += 1
bgm /= np.abs(bgm).max()

# gain automation: duck under voice, silence at the crack, fade in/out
tt = np.arange(N) / SR
g = np.full(N, 0.30)
for a, b in vo_spans:
    i, j = at(a - 0.15), at(b + 0.2)
    g[i:j] = 0.13
g = np.convolve(g, np.ones(int(0.12 * SR)) / int(0.12 * SR), mode="same")
crack = next(s for s in tl if s["vo"] == "crack")
tc = crack["start"] + crack["vo_at"] + 0.62 * crack["vo_len"]
proud = next(s for s in tl if s["vo"] == "proud")
mute = np.ones(N)
m = (tt > tc - 0.05) & (tt < proud["start"])
mute[m] = 0.0
ramp = (tt >= proud["start"]) & (tt < proud["start"] + 1.0)
mute[ramp] = (tt[ramp] - proud["start"]) / 1.0
g *= mute
g *= np.minimum(1, tt / 0.8) * np.clip((total - tt) / 1.8, 0, 1)
bgm *= g

# ------------------------------------------------------------------ SFX
sfx = np.zeros(N)


def scribble(dur):
    x = bandnoise(dur, 2500, 7000)
    t = np.arange(len(x)) / SR
    am = 0.5 + 0.5 * np.sin(2 * np.pi * (9 + 3 * np.sin(t * 2.1)) * t) ** 2
    return x * am * np.minimum(1, t / 0.05) * np.minimum(1, (dur - t) / 0.1)


def pop():
    t = np.arange(int(0.12 * SR)) / SR
    f = 500 + 900 * t / 0.12
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 30)


def whoosh():
    x = bandnoise(0.35, 300, 2500)
    t = np.arange(len(x)) / SR
    return x * np.sin(np.pi * t / 0.35) ** 2


def crack_sfx():
    x = bandnoise(0.5, 800, 9000)
    t = np.arange(len(x)) / SR
    snap = x * np.exp(-t * 22)
    thump = np.sin(2 * np.pi * 70 * t) * np.exp(-t * 10)
    return snap + 0.8 * thump


S = {s["type"]: s for s in tl}
for s in tl[1:]:
    add(sfx, whoosh(), s["start"] - 0.15, 0.25)
for typ, a, b in [("Hook", 0.2, 1.8), ("Centering", 0.1, 1.2), ("Kiln", 0.0, 1.0), ("Crack", 0.0, 1.2),
                  ("Outro", 0.0, 1.4), ("ClayPhoto", 0.9, 1.6)]:
    add(sfx, scribble(b - a), S[typ]["start"] + a, 0.10)
pops = [("Hook", 1.5), ("Hook", 1.8), ("Hook", 2.1), ("Proud", 0.3), ("Trimming", 2.3 + 0.7),
        ("Centering", 1.0), ("Glaze", 0.4), ("Glaze", 0.95), ("Glaze", 1.5)]
for typ, tt0 in pops:
    add(sfx, pop(), S[typ]["start"] + tt0, 0.22)
add(sfx, crack_sfx(), tc, 0.7)
cen = S["Centering"]
add(sfx, crack_sfx() * 0.4, cen["start"] + 0.74 * cen["dur"], 0.5)   # clay slump thud

mixv = voice * 0.95 + bgm + sfx
mixv = np.tanh(mixv * 1.1) / np.tanh(1.1)
mixv = mixv[: int(total * SR)]
wav = os.path.splitext(out)[0] + "_audio.wav"
sf.write(wav, np.c_[mixv, mixv].astype(np.float32), SR)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart", out], check=True)
print("wrote", out, "%.1fs" % total)
