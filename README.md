# 🎙️ AudioScene Analyzer

> **Speech transcription · Emotion recognition · Background sound detection → structured JSON**

A Google Colab notebook that performs end-to-end audio intelligence on any audio file. It combines four specialized models into a single pipeline and outputs a clean, segment-level `scene.json` you can feed directly into downstream applications.

---

## What it does

For every spoken segment in your audio the pipeline produces:

| Field | Description |
|---|---|
| `start` / `end` | Timestamp in seconds |
| `text` | Transcribed speech |
| `speaker` | Speaker label (e.g. `SPEAKER_00`) |
| `emotion` | Detected emotion (`angry`, `happy`, `sad`, `neutral`, …) |
| `background_scene` | Dominant background sound at that moment |
| `words` | Word-level timestamps |

---

## Model stack

| Model | Task |
|---|---|
| [WhisperX](https://github.com/m-bain/whisperX) (`large-v2`) | Transcription, word-level alignment, speaker diarization |
| [PANNs](https://github.com/qiuqiangkong/panns_inference) | Clip-level audio tagging + frame-level sound event detection |
| [HuBERT](https://huggingface.co/superb/hubert-large-superb-er) (`superb/hubert-large-superb-er`) | Per-segment emotion recognition |
| [pyannote](https://github.com/pyannote/pyannote-audio) | Speaker diarization (via WhisperX) |

---

## Quick start

### 1. Open in Colab

Upload `src/audioscene_analyzer.ipynb` to [Google Colab](https://colab.research.google.com) and select a **GPU runtime** (Runtime → Change runtime type → T4 GPU).

### 2. Set your HuggingFace token *(for diarization)*

Diarization requires accepting the pyannote model license on HuggingFace and providing a token.

1. Go to [hf.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token.
2. Accept the [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) model conditions.
3. In Colab, open **Secrets** (🔑 icon) and add `HF_TOKEN` = your token.

### 3. Point to your audio file

```python
AUDIO_PATH = "/content/drive/MyDrive/your_audio.mp3"  # MP3, WAV, FLAC, etc.
```

Mount Google Drive first if needed:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Run all cells

The notebook installs dependencies automatically on first run. Processing time scales with audio length and GPU tier.

---

## Output

Results are saved to `outputs/scene.json`. Example segment:

```json
{
  "start": 4.521,
  "end": 8.743,
  "text": "I can't believe this happened.",
  "speaker": "SPEAKER_00",
  "emotion": "angry",
  "background_scene": "Music",
  "words": [
    { "word": "I", "start": 4.521, "end": 4.62 },
    { "word": "can't", "start": 4.62, "end": 4.88 }
  ]
}
```

---

## Pipeline overview

```
Audio file
    │
    ├─► librosa (32 kHz) ──► PANNs AudioTagging   → clip-level tags
    │                    ──► PANNs SoundEventDet.  → frame-level background
    │
    ├─► librosa (16 kHz) ──► HuBERT               → emotion per segment
    │
    └─► WhisperX ──────────► Transcription
                         ──► Word alignment
                         ──► Diarization (pyannote)
                              │
                              └─────────────────────► scene.json ✅
```

---

## Requirements

All dependencies are installed automatically by the notebook's first cell:

```
torch · torchaudio · librosa · panns-inference
transformers · soundfile · whisperx · ffmpeg
```

**Runtime:** Google Colab with GPU (CUDA). CPU fallback works but is significantly slower; `float16` is used on GPU, `int8` on CPU.

---

## Configuration

Key variables at the top of the notebook:

```python
AUDIO_PATH    = "/content/drive/MyDrive/test.mp3"   # path to your audio
HF_TOKEN      = ""        # HuggingFace token for diarization
SCORE_THRESHOLD = 0.3     # minimum confidence for background sound events
SEGMENT_LEN_SEC = 1.0     # background detection window in seconds
```

---

## Notes & limitations

- Diarization is **optional** — skipped if `HF_TOKEN` is not set; `speaker` defaults to `"unknown"`.
- Very short clips (< 1000 samples) are tagged `"unknown"` for emotion.
- Background sound matching uses a simple overlap heuristic; overlapping scenes pick the highest-confidence label.
- WhisperX `large-v2` requires ~10 GB VRAM; use `medium` or `small` if you hit memory errors.

---

## License

MIT
