# bia-music-composer

A personal fine-tuning pipeline for [HeartMuLa](https://github.com/HeartMuLa/heartlib) — an open-source music foundation model — trained on paired audio examples to compose music the way I do.

My name is Bianca (bia). I've spent my life making music across many different bands and projects, each with its own sound and intention. This repository is an attempt to distill that accumulated voice into a model that can generate new music in that spirit. Not to replace the creative act — but to have a collaborator that has listened deeply.

---

## What this is

HeartMuLa is a 3B–7B parameter language model for music. It tokenizes audio with HeartCodec (a neural codec running at 12.5 Hz), generates token sequences conditioned on lyrics and style tags, and decodes them back to full-quality waveforms. The architecture separates global song planning from local rhythmic and timbral detail — which is what makes long-form generation coherent.

This project wraps HeartMuLa with a fine-tuning and inference pipeline that teaches it to compose music from *my* paired audio examples: recordings I've made, covers I've arranged, demos I've tracked over the years. Given a new input recording, the fine-tuned model generates music that sounds like it belongs to the same body of work.

---

## How it works

### Data format

Paired audio lives in `data/paired/`:

```
data/paired/
  inputs/    ← source audio (e.g. a rough demo, a reference recording)
  outputs/   ← target audio (e.g. the finished arrangement, my version)
  tags/      ← optional style tags per stem (auto-generated if absent)
  lyrics/    ← optional lyrics per stem (auto-generated if absent)
```

Files are matched by stem name: `001.wav` in `inputs/` pairs with `001.wav` in `outputs/`.

### Auto-annotation

[scripts/annotate.py](scripts/annotate.py) automatically generates tags and lyrics for any input audio using:

- **LAION CLAP** (`laion/clap-htsat-unfused`) — zero-shot audio classification across genre, instrument, mood, tempo, and language
- **OpenAI Whisper large-v3** — lyric transcription with automatic language detection

This means you can drop in raw audio without any manual tagging.

### Fine-tuning approaches

Two training scripts are provided, each with a different inductive bias:

**[scripts/finetune_paired.py](scripts/finetune_paired.py)** — Importance-weighted proxy training. Since HeartCodec does not expose a public audio encoder, this approach generates candidate code sequences from the base model, decodes them, computes mel-spectrogram cosine similarity to the real target WAV, and trains with sample weights proportional to that similarity. Slower but useful when direct reference conditioning is insufficient.

**[scripts/finetune_audio2audio.py](scripts/finetune_audio2audio.py)** — WavLM-conditioned generation with a learned `AudioConditioningModule`. WavLM frame features from the input audio are adaptively pooled to `NUM_PREFIX_TOKENS` conditioning vectors, projected to backbone dimension, and injected as an additive bias into every backbone forward pass. This carries melodic/timbral information from the input without touching sequence length, positional embeddings, or KV caches.

> Aside, AudioConditioningModule bridges WavLM and HeartMuLa's backbone.
> 
> WavLM encodes input audio into a sequence of frame-level feature vectors (e.g. 768-dimensional, one per ~20ms of audio). That's potentially hundreds of vectors for a 30-second clip — too long and the wrong shape to feed directly into HeartMuLa's backbone.
> 
> AudioConditioningModule does three things:
> 
> Pools the WavLM frames down to exactly NUM_PREFIX_TOKENS (default 32) time steps using adaptive average pooling — preserving temporal ordering (verse→chorus shape) while compressing the sequence to a fixed length
Projects each of those 32 vectors from WavLM's dimension (768) to the backbone's hidden dimension
Adds learned positional embeddings so the backbone knows the ordering of those conditioning slots
The result is a tensor of shape (1, 32, backbone_dim). Rather than prepending these as extra tokens (which would break RoPE and KV-cache position tracking), the 32 vectors are averaged into a single global conditioning vector and added as a bias to every hidden state at every backbone forward call via a hook.
> 
> In plain terms: it takes "what does this input recording sound like" and quietly nudges every generation step toward that sonic character, without changing the sequence length or the model's positional reasoning.

Both use [LoRA](https://github.com/huggingface/peft) (rank 16 by default) on the attention projections (`q_proj`, `k_proj`, `v_proj`, `output_proj`) of the HeartMuLa global backbone.

### Inference

**[scripts/generate.py](scripts/generate.py)** — Loads the base model, applies the trained LoRA adapter, auto-annotates the input WAV, and generates N clips conditioned on the reference audio. Clips are saved locally and uploaded to GCS.

**[scripts/generate_audio2audio.py](scripts/generate_audio2audio.py)** — The same, but also loads the trained `AudioConditioningModule` and installs WavLM-derived prefix conditioning into the backbone via a forward hook before generation.

---

## Setup

### Requirements

- Python 3.12+
- CUDA 12.8 (for GPU training — CPU inference is slow but possible)
- [uv](https://github.com/astral-sh/uv) for dependency management
- A GCS bucket (for model caching and checkpoint storage)
- `gsutil` and `gcloud` CLI tools

### Install

```bash
git clone https://github.com/biancachengcostanzo/bia-music-composer.git
cd bia-music-composer
uv sync
```

### Environment variables

Copy `.env.example` to `.env` and fill in:

```env
GCS_BUCKET_NAME=your-bucket-name
GCS_BUCKET_FOLDER_PREFIX=heartmula
PROJECT_ID=your-gcp-project-id
DOCKER_IMAGE=gcr.io/your-project/bia-music-composer

MODEL_SIZE=3b
RUN_MODE=train

NUM_EPOCHS=3
LEARNING_RATE=5e-5
LORA_RANK=16
MAX_AUDIO_MS=30000

AUDIO_ENCODER_MODEL=m-a-p/MERT-v1-95M
NUM_PREFIX_TOKENS=32
```

---

## Running locally

### Annotate a file

```bash
uv run python scripts/annotate.py data/paired/inputs/001.wav
# Writes data/paired/tags/001.txt and data/paired/lyrics/001.txt
```

### Fine-tune (audio-to-audio LoRA)

```bash
GCS_BUCKET_NAME=your-bucket uv run python scripts/finetune.py
```

### Generate with a fine-tuned adapter

```bash
INPUT_WAV=data/paired/inputs/001.wav \
GCS_BUCKET_NAME=your-bucket \
uv run python scripts/generate.py
# Outputs: out/generated/generated_001.wav, _002.wav, _003.wav
```

---

## Running on GCP

The `gcp/` directory contains shell scripts to launch SPOT VMs for training and inference. SPOT VMs are preemptible but inexpensive — the scripts include GCS checkpointing so training resumes automatically on restart.

```bash
# Fine-tune with WavLM conditioning on a SPOT L4 GPU
bash gcp/run_finetune_audio2audio.sh

# Generate clips
bash gcp/run_generate_audio2audio.sh
```

Monitor logs:

```bash
gcloud logging tail 'resource.type=gce_instance' \
  --project=$PROJECT_ID \
  --format='value(jsonPayload.message)'
```

The Docker image is built and pushed to GCR with:

```bash
bash gcp/build_and_push.sh
```

---

## Model weights

Base model weights are downloaded automatically from HuggingFace on first run (or from a GCS cache if configured):

- `HeartMuLa/HeartMuLaGen` — shared tokenizer and generation assets
- `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` — 3B backbone weights (~16 GB)
- `HeartMuLa/HeartCodec-oss-20260123` — audio codec

Fine-tuned LoRA adapters and AudioConditioningModule weights are stored in GCS under `gs://$GCS_BUCKET_NAME/$GCS_BUCKET_FOLDER_PREFIX-3b/$RUN_MODE/latest/`.

---

## Project structure

```
scripts/
  annotate.py                  auto-tag and transcribe audio
  finetune_paired.py           importance-weighted proxy fine-tuning
  finetune_audio2audio.py      WavLM-conditioned LoRA fine-tuning
  finetune_audio2audio_direct.py  direct variant
  generate.py                  inference with LoRA adapter
  generate_audio2audio.py      inference with WavLM prefix conditioning

gcp/
  build_and_push.sh            build and push Docker image to GCR
  run_finetune_audio2audio.sh  launch SPOT VM for audio2audio fine-tune
  run_generate_audio2audio.sh  launch SPOT VM for audio2audio generation
  run_generate.sh              launch SPOT VM for standard generation
  finetune_audio2audio_startup.sh  VM startup script
  generate_audio2audio_startup.sh  VM startup script
  run_and_watch.sh             launch and tail logs

Dockerfile                     CUDA 12.8 + uv + gcloud SDK image
pyproject.toml                 Python project definition
```

---

## Acknowledgments

Built on top of [HeartMuLa](https://github.com/HeartMuLa/heartlib), an open-source music foundation model from the HeartMuLa team. The paper, architecture, and production code are unusually well-aligned — which is what made this personal project tractable. With thanks to [@christian-bick](https://www.github.com/christian-bick) for the ML training crash course.

---

## License

Apache 2.0 — see [LICENSE](LICENSE). This project builds on [HeartMuLa](https://github.com/HeartMuLa/heartlib), which is also licensed under Apache 2.0.
