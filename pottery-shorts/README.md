# 도자기 취미 쇼츠

Scripts that build a 1080x1920 YouTube Shorts video about a pottery hobby: hand-drawn
animation scenes, the user's photos and clips with doodle overlays, Korean narration
(offline Supertonic 3 TTS via sherpa-onnx), captions, and a synthesized BGM with SFX.

- `narration.py <model_dir> <vo_dir> [speaker_id]`: generate narration clips
- `render.py <vo_dir> <video_silent.mp4>`: render the frames (`--stills idx:t ...` saves preview stills)
- `mix.py <vo_dir> <video_silent.mp4> <out.mp4>`: mix the audio and mux it with the video

Personal photos and videos (`assets/media/`) and the rendered output are not committed.
Fonts: Nanum Pen Script and Gaegu (SIL OFL).
