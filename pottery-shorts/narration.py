"""Generate Korean narration clips with the offline Supertonic 3 model (via sherpa-onnx)."""
import json, os, sys
import numpy as np
import sherpa_onnx as s
import soundfile as sf

MODEL = sys.argv[1]          # folder with the reassembled Supertonic files
OUT = sys.argv[2]            # output folder for wav clips
SID = int(sys.argv[3]) if len(sys.argv) > 3 else 1

LINES = {
    "hook":    "도자기 컵 하나 만드는 데, 얼마나 걸릴 것 같아요?",
    "clay":    "시작은 이렇게, 투박한 흙 한 덩어리예요.",
    "center":  "물레 위에서 중심 잡기부터가 첫 번째 고비. 힘을 조금만 놓쳐도, 흙이 비틀비틀 무너져요.",
    "dry":     "겨우 모양을 만들면, 꾸덕하게 마를 때까지 또 기다려야 하고요.",
    "trim":    "다 마르면 뒤집어서 굽을 깎아요. 깎아낸 흙이 이만큼이에요.",
    "vase":    "손잡이를 붙이고, 스펀지로 다듬고. 손이 몇십 번은 가요.",
    "kiln":    "그다음엔 초벌, 유약, 재벌. 가마에만 두 번 들어갔다 나와야 해요.",
    "glaze":   "원하는 색 하나 찾으려고, 유약 테스트도 몇 번씩 하고요.",
    "crack":   "그렇게 한 달을 기다렸는데, 가마를 열면 금이 가 있을 때도 있어요.",
    "proud":   "그래도, 내 손으로 끝까지 완성한 걸 처음 손에 쥐는 순간.",
    "proud2":  "그 뿌듯함 때문에, 저는 또 물레 앞에 앉아요.",
    "outro":   "제 취미는, 도자기 만들기예요.",
}

m = MODEL.rstrip("/") + "/"
sc = s.OfflineTtsSupertonicModelConfig(
    m + "duration_predictor.onnx", m + "text_encoder.onnx", m + "vector_estimator.onnx",
    m + "vocoder.onnx", m + "tts.json", m + "unicode_indexer.bin", m + "voice.bin")
mc = s.OfflineTtsModelConfig()
mc.supertonic = sc
mc.num_threads = 4
tts = s.OfflineTts(s.OfflineTtsConfig(model=mc))

os.makedirs(OUT, exist_ok=True)
durations = {}
for key, text in LINES.items():
    g = s.GenerationConfig()
    g.sid = SID
    g.num_steps = 16
    g.speed = 1.05
    g.extra = {"lang": "ko"}
    a = tts.generate(text, g)
    x = np.array(a.samples, dtype=np.float32)
    # trim leading/trailing near-silence
    thr = 0.01
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx):
        x = x[max(0, idx[0] - 800): idx[-1] + 2000]
    x = x / (np.abs(x).max() + 1e-6) * 0.9
    sf.write(f"{OUT}/{key}.wav", x, a.sample_rate)
    durations[key] = len(x) / a.sample_rate
    print(f"{key:8s} {durations[key]:.2f}s  {text}")

json.dump({"sr": a.sample_rate, "durations": durations, "lines": LINES},
          open(f"{OUT}/narration.json", "w"), ensure_ascii=False, indent=1)
