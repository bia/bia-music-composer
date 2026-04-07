#!/usr/bin/env python3
"""
Few-shot fine-tune HeartMuLa on paired audio examples (input → output).

Approach
--------
HeartCodec does not expose a public audio encoder, so we cannot directly
convert real WAV files into discrete training targets.

Instead we use importance-weighted proxy training:
  1.  For each output WAV, generate NUM_PROXY_SAMPLES candidate code sequences
      from the base model using the pair's style tags.
  2.  Decode each candidate to a waveform via pipe.codec.decode() and compute
      mel-spectrogram cosine similarity to the real target WAV.
  3.  Normalise the similarities into sample weights.
  4.  Train with teacher-forcing cross-entropy loss on all proxy samples,
      each weighted by how closely it resembles the real target.

The result: LoRA adapters that steer the model toward generating audio
that sounds like your real target, without needing a codec encoder.

Directory convention
--------------------
    data/paired/outputs/  ← target WAV files  (stem = pair id, e.g. 001.wav)
    data/paired/inputs/   ← optional input WAVs (currently used for reference;
                            not yet encoded as conditioning tokens)
    data/paired/tags/     ← optional per-pair style tags  (stem.txt)

Required env vars
-----------------
    GCS_BUCKET_NAME

Optional env vars
-----------------
    PAIRS_DIR           Root of the paired audio folder  (default: ./data/paired)
    MODEL_SIZE          HeartMuLa variant                (default: "3b")
    STYLE_TAGS          Fallback comma-separated tags    (default: "piano,ambient,reflective")
    NUM_PROXY_SAMPLES   Candidates to generate per pair  (default: 3)
    NUM_EPOCHS          Training epochs                  (default: 5)
    LEARNING_RATE       AdamW LR                         (default: 3e-5)
    LORA_RANK           LoRA rank                        (default: 16)
    MAX_AUDIO_MS        Max audio length per sample (ms) (default: 30000)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

try:
    import heartlib  # noqa: F401
except ImportError:
    print("heartlib not found — installing from GitHub…")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps",
         "git+https://github.com/HeartMuLa/heartlib.git"],
        check=True,
    )

from heartlib import HeartMuLaGenPipeline  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CODEBOOKS = 8
EMPTY_ID      = 0
CODEC_FPS     = 12.5   # HeartCodec frames per second

# ── Config ─────────────────────────────────────────────────────────────────────
GCS_BUCKET_NAME          = os.environ["GCS_BUCKET_NAME"]
GCS_BUCKET_FOLDER_PREFIX = os.getenv("GCS_BUCKET_FOLDER_PREFIX", "heartmula")
MODEL_SIZE               = os.getenv("MODEL_SIZE", "3b").upper()
RUN_MODE                 = os.getenv("RUN_MODE", "paired")
PAIRS_DIR                = Path(os.getenv("PAIRS_DIR", "./data/paired"))
STYLE_TAGS               = os.getenv("STYLE_TAGS", "piano,ambient,reflective")
NUM_PROXY_SAMPLES        = int(os.getenv("NUM_PROXY_SAMPLES", "3"))
NUM_EPOCHS               = int(os.getenv("NUM_EPOCHS",        "5"))
LEARNING_RATE            = float(os.getenv("LEARNING_RATE",   "3e-5"))
LORA_RANK                = int(os.getenv("LORA_RANK",         "16"))
LORA_ALPHA               = LORA_RANK * 2
MAX_AUDIO_MS             = int(os.getenv("MAX_AUDIO_MS",      "30000"))

CKPT_DIR           = Path("./ckpt")
OUTPUT_ADAPTER_DIR = Path("./out/adapters/paired")
OUTPUT_MODEL_DIR   = Path("./out/models/paired")
TMP_DIR            = Path("./data/tmp")

GCS_MODEL_CACHE = f"gs://{GCS_BUCKET_NAME}/model-cache"
_GCS_RUN_BASE   = (f"gs://{GCS_BUCKET_NAME}/{GCS_BUCKET_FOLDER_PREFIX}"
                   f"-{MODEL_SIZE}/{RUN_MODE}/run-checkpoint")
LORA_CKPT_GCS   = f"{_GCS_RUN_BASE}/lora_adapter"
EPOCH_CKPT_GCS  = f"{_GCS_RUN_BASE}/epoch.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

print(f"\nDevice : {DEVICE}")
print(f"Model  : HeartMuLa-oss-{MODEL_SIZE}")
print(f"Style  : tags='{STYLE_TAGS}' (per-pair tags override if present)")
print(f"Train  : {NUM_PROXY_SAMPLES} proxy samples × {NUM_EPOCHS} epochs  "
      f"lr={LEARNING_RATE}  LoRA rank={LORA_RANK}\n")


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gcs_dir_exists(gcs_path: str) -> bool:
    return subprocess.run(
        ["gsutil", "ls", gcs_path], capture_output=True
    ).returncode == 0


def _gcs_cp(src: str, dst: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", "-r", src, dst], check=True)


def _gcs_upload(local: str, gcs: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", "-r", local, gcs], check=True)


def _gcs_download(gcs: str, local: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", "-r", gcs, local], check=True)


# ── 1. Download model checkpoints ─────────────────────────────────────────────

def download_checkpoints() -> None:
    mula_dir  = CKPT_DIR / f"HeartMuLa-oss-{MODEL_SIZE}"
    codec_dir = CKPT_DIR / "HeartCodec-oss"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not (CKPT_DIR / "tokenizer.json").exists():
        gcs_src = f"{GCS_MODEL_CACHE}/HeartMuLaGen"
        if _gcs_dir_exists(gcs_src):
            print("Downloading HeartMuLaGen assets from GCS cache…")
            _gcs_cp(f"{gcs_src}/*", str(CKPT_DIR) + "/")
        else:
            subprocess.run(
                ["huggingface-cli", "download", "--local-dir", str(CKPT_DIR),
                 "HeartMuLa/HeartMuLaGen"], check=True)

    if not mula_dir.exists():
        gcs_src = f"{GCS_MODEL_CACHE}/HeartMuLa-oss-{MODEL_SIZE}"
        if _gcs_dir_exists(gcs_src):
            print(f"Downloading HeartMuLa-oss-{MODEL_SIZE} from GCS cache…")
            _gcs_cp(gcs_src, str(CKPT_DIR) + "/")
        else:
            subprocess.run(
                ["huggingface-cli", "download", "--local-dir", str(mula_dir),
                 "HeartMuLa/HeartMuLa-oss-3B-happy-new-year"], check=True)

    if not codec_dir.exists():
        gcs_src = f"{GCS_MODEL_CACHE}/HeartCodec-oss"
        if _gcs_dir_exists(gcs_src):
            print("Downloading HeartCodec-oss from GCS cache…")
            _gcs_cp(gcs_src, str(CKPT_DIR) + "/")
        else:
            subprocess.run(
                ["huggingface-cli", "download", "--local-dir", str(codec_dir),
                 "HeartMuLa/HeartCodec-oss-20260123"], check=True)


# ── 2. Load pipeline ───────────────────────────────────────────────────────────

def load_pipeline() -> HeartMuLaGenPipeline:
    print("Loading HeartMuLa pipeline…")
    return HeartMuLaGenPipeline.from_pretrained(
        pretrained_path=str(CKPT_DIR),
        device={"mula": DEVICE, "codec": DEVICE},
        dtype={"mula": DTYPE, "codec": DTYPE},
        version=MODEL_SIZE,
    )


# ── 3. Discover paired audio files ────────────────────────────────────────────

def load_pairs() -> list[dict]:
    """
    Scan PAIRS_DIR/outputs/ for target WAVs.  Optionally matches inputs/ and
    data/paired/tags/{stem}.txt for per-pair tags.
    """
    outputs_dir = PAIRS_DIR / "outputs"
    inputs_dir  = PAIRS_DIR / "inputs"
    tags_dir    = Path(os.getenv("TAGS_DIR", "./data/paired/tags"))

    if not outputs_dir.exists():
        raise FileNotFoundError(
            f"Expected directory: {outputs_dir}\n"
            "Create it and populate with target WAV files."
        )

    pairs = []
    for output_path in sorted(outputs_dir.glob("*.wav")):
        s = output_path.stem
        tags_file = tags_dir / f"{s}.txt"
        tags = tags_file.read_text().strip() if tags_file.exists() else STYLE_TAGS
        input_path = inputs_dir / f"{s}.wav" if inputs_dir.exists() else None
        pairs.append({
            "stem":   s,
            "output": output_path,
            "input":  input_path if (input_path and input_path.exists()) else None,
            "tags":   tags,
        })

    print(f"Found {len(pairs)} audio pair(s):")
    for p in pairs:
        has_input = "✓" if p["input"] else "–"
        print(f"  {p['stem']}  input={has_input}  tags='{p['tags']}'")
    return pairs


# ── 4. Load target WAVs as mel-spectrograms ───────────────────────────────────

MEL_SR    = 22050   # resample target WAVs to this for mel computation
MEL_XFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=MEL_SR, n_fft=1024, hop_length=256, n_mels=80
)


def load_mel(wav_path: Path) -> torch.Tensor:
    """Load a WAV and return its log-mel-spectrogram (80, T)."""
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != MEL_SR:
        waveform = torchaudio.functional.resample(waveform, sr, MEL_SR)
    mel = MEL_XFORM(waveform.squeeze(0))           # (80, T)
    return (mel + 1e-6).log()


def mel_similarity(mel1: torch.Tensor, mel2: torch.Tensor) -> float:
    """Cosine similarity between two log-mel-spectrograms of potentially different length."""
    min_t = min(mel1.shape[-1], mel2.shape[-1])
    v1 = mel1[..., :min_t].flatten()
    v2 = mel2[..., :min_t].flatten()
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


# ── 5. Generate proxy code sequences ──────────────────────────────────────────

def _write_prompt_files(tags: str) -> dict:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tags_file   = TMP_DIR / "tags.txt"
    lyrics_file = TMP_DIR / "lyrics.txt"
    tags_file.write_text(tags.lower())
    lyrics_file.write_text("")
    return {"tags": str(tags_file), "lyrics": str(lyrics_file)}


def generate_proxy_codes(pipe: HeartMuLaGenPipeline, tags: str) -> torch.Tensor:
    """
    Run one inference pass and return the generated discrete audio codes.
    Returns shape (8, T_frames).
    """
    inputs = _write_prompt_files(tags)
    try:
        model_inputs = pipe.preprocess(inputs, cfg_scale=1.0)
    except TypeError:
        model_inputs = pipe.preprocess(inputs)

    with torch.no_grad():
        try:
            model_outputs = pipe._forward(
                model_inputs,
                max_audio_length_ms=MAX_AUDIO_MS,
                temperature=1.0,
                topk=50,
                cfg_scale=1.0,
            )
        except TypeError:
            model_outputs = pipe._forward(model_inputs, max_audio_length_ms=MAX_AUDIO_MS)

    return model_outputs["frames"], model_inputs["tokens"]   # (8, T), (1|2, T_text, 9)


def decode_codes_to_mel(pipe: HeartMuLaGenPipeline, frames: torch.Tensor) -> torch.Tensor:
    """
    Decode discrete codes to a waveform via pipe.codec.decode(), then compute
    log-mel-spectrogram.  Returns (80, T_mel).
    """
    with torch.no_grad():
        # postprocess expects the model_outputs dict format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        pipe.postprocess({"frames": frames}, tmp_path)
        waveform, sr = torchaudio.load(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != MEL_SR:
        waveform = torchaudio.functional.resample(waveform, sr, MEL_SR)
    mel = MEL_XFORM(waveform.squeeze(0))
    return (mel + 1e-6).log()


# ── 6. Build teacher-forcing sequence ─────────────────────────────────────────

def build_training_sequence(
    prompt_tokens: torch.Tensor,   # (1, T_text, 9)
    audio_codes: torch.Tensor,     # (8, T_audio)
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Assemble token tensor for teacher-forcing.

    Token tensor columns (dim -1 = 9):
        cols 0–7 : codebook indices (EMPTY_ID at text positions)
        col  8   : text token id   (EMPTY_ID at audio positions)

    Returns tokens (1, T_total, 9), mask (1, T_total, 9), T_text.
    """
    if prompt_tokens.shape[0] == 2:   # drop CFG unconditional copy
        prompt_tokens = prompt_tokens[:1]

    T_text  = prompt_tokens.shape[1]
    T_audio = audio_codes.shape[1]
    T_total = T_text + T_audio

    tokens = torch.full(
        (1, T_total, NUM_CODEBOOKS + 1), EMPTY_ID,
        dtype=torch.long, device=DEVICE,
    )
    mask = torch.zeros(
        (1, T_total, NUM_CODEBOOKS + 1),
        dtype=torch.bool, device=DEVICE,
    )

    tokens[0, :T_text, :]               = prompt_tokens[0].to(DEVICE)
    mask[0,   :T_text, -1]              = True
    tokens[0, T_text:, :NUM_CODEBOOKS]  = audio_codes.T.long().to(DEVICE)
    mask[0,   T_text:, :NUM_CODEBOOKS]  = True

    return tokens, mask, T_text


# ── 7. Loss function ───────────────────────────────────────────────────────────

def _backbone_hidden_states(
    mula,
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    embeds = mula._embed_tokens(tokens, uncond_mask=None)               # (1, S, 9, D)
    h = (embeds * mask.unsqueeze(-1)).sum(dim=2).to(DTYPE)              # (1, S, D)
    S = h.shape[1]
    causal_mask = torch.tril(
        torch.ones(S, S, dtype=torch.bool, device=DEVICE)
    ).unsqueeze(0)
    return mula.backbone(h, mask=causal_mask)                           # (1, S, D)


def compute_loss(
    mula,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    T_text: int,
) -> torch.Tensor:
    """
    Next-token cross-entropy on codebook-0 at audio positions.
    """
    h = _backbone_hidden_states(mula, tokens[:, :-1, :], mask[:, :-1, :])
    audio_h = h[:, T_text - 1:, :].to(DTYPE)
    logits  = mula.codebook0_head(audio_h)
    targets = tokens[:, T_text:, 0]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


# ── 8. LoRA setup ─────────────────────────────────────────────────────────────

def disable_backbone_caches(mula) -> None:
    bb = mula.backbone
    for module in bb.modules():
        if hasattr(module, "kv_cache"):
            module.kv_cache = None
    if hasattr(bb, "_validate_inputs"):
        bb._validate_inputs = lambda *args, **kwargs: None


def apply_lora(mula) -> None:
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(mula.backbone, lora_cfg)
    mula.backbone    = peft_model
    mula._peft_model = peft_model
    peft_model.print_trainable_parameters()


# ── 9. Training loop ───────────────────────────────────────────────────────────

def _load_start_epoch() -> int:
    result = subprocess.run(
        ["gsutil", "cat", EPOCH_CKPT_GCS], capture_output=True, text=True)
    if result.returncode == 0:
        epoch = int(result.stdout.strip())
        print(f"Resuming: starting from epoch {epoch + 1}.")
        return epoch + 1
    return 0


def train_loop(mula, training_samples: list) -> None:
    """
    training_samples : list of dicts with keys:
        tokens  LongTensor (1, T_total, 9)
        mask    BoolTensor (1, T_total, 9)
        T_text  int
        weight  float  — mel-similarity weight (higher = closer to real target)
    """
    for p in mula.codebook0_head.parameters():
        p.requires_grad_(True)

    trainable  = [p for p in mula.backbone.parameters() if p.requires_grad]
    trainable += list(mula.codebook0_head.parameters())
    optimizer  = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

    start_epoch = _load_start_epoch()
    if start_epoch > 0 and _gcs_dir_exists(LORA_CKPT_GCS):
        print("Resuming: loading LoRA adapter from GCS checkpoint…")
        local_lora_parent = Path("./ckpt")
        local_lora = local_lora_parent / "lora_adapter"
        _gcs_download(LORA_CKPT_GCS, str(local_lora_parent) + "/")
        mula._peft_model.load_adapter(str(local_lora), adapter_name="default")

    mula.train()
    total_weight = sum(s["weight"] for s in training_samples)

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for sample in training_samples:
            loss = compute_loss(mula, sample["tokens"], sample["mask"], sample["T_text"])
            # Weight by normalised mel similarity so samples closer to real target
            # contribute more to the gradient
            scaled = loss * (sample["weight"] / total_weight) * len(training_samples)
            scaled.backward()
            epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        avg = epoch_loss / len(training_samples)
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS}  loss={avg:.4f}")

        # Checkpoint
        local_lora = Path("./ckpt/lora_ckpt_paired")
        mula._peft_model.save_pretrained(str(local_lora))
        _gcs_upload(str(local_lora), LORA_CKPT_GCS)
        subprocess.run(
            ["gsutil", "cp", "-", EPOCH_CKPT_GCS],
            input=str(epoch), text=True, check=True,
        )
        print(f"  Checkpoint saved after epoch {epoch + 1}.")

    mula.eval()


# ── 10. Save outputs ───────────────────────────────────────────────────────────

def save_outputs(mula) -> None:
    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mula._peft_model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    torch.save(mula.codebook0_head.state_dict(),
               OUTPUT_ADAPTER_DIR / "codebook0_head.pt")
    print(f"LoRA adapter  → {OUTPUT_ADAPTER_DIR}")

    merged = mula._peft_model.merge_and_unload()
    torch.save(merged.state_dict(), OUTPUT_MODEL_DIR / "model.pt")
    torch.save(mula.codebook0_head.state_dict(),
               OUTPUT_MODEL_DIR / "codebook0_head.pt")
    print(f"Merged model  → {OUTPUT_MODEL_DIR}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    download_checkpoints()
    pipe = load_pipeline()

    pairs = load_pairs()

    # Phase 1: generate proxy code sequences and score them against real targets
    print(f"\nGenerating {NUM_PROXY_SAMPLES} proxy sample(s) per pair "
          f"and scoring against real target audio…")

    training_samples = []
    for pair in pairs:
        target_mel = load_mel(pair["output"])
        print(f"\n  Pair {pair['stem']}  (target: {pair['output'].name})")

        candidates = []
        for i in range(NUM_PROXY_SAMPLES):
            print(f"    proxy {i + 1}/{NUM_PROXY_SAMPLES} …", end=" ", flush=True)
            frames, prompt_tokens = generate_proxy_codes(pipe, pair["tags"])
            proxy_mel  = decode_codes_to_mel(pipe, frames)
            similarity = mel_similarity(target_mel, proxy_mel)
            print(f"{frames.shape[1] / CODEC_FPS:.1f}s  sim={similarity:.3f}")
            candidates.append((frames, prompt_tokens, similarity))

        # Normalise similarities to weights (softmax for stability)
        sims   = torch.tensor([c[2] for c in candidates])
        weights = torch.softmax(sims * 5.0, dim=0).tolist()   # temperature=5 sharpens weights

        for (frames, prompt_tokens, sim), w in zip(candidates, weights):
            tokens, mask, T_text = build_training_sequence(prompt_tokens, frames)
            training_samples.append({
                "tokens": tokens,
                "mask":   mask,
                "T_text": T_text,
                "weight": w,
                "sim":    sim,
                "stem":   pair["stem"],
            })
            print(f"    → weight={w:.3f}  sim={sim:.3f}")

    # Phase 2: fine-tune with importance-weighted teacher-forcing
    print("\nApplying LoRA to backbone…")
    disable_backbone_caches(pipe.mula)
    apply_lora(pipe.mula)

    print(f"\nFine-tuning for {NUM_EPOCHS} epochs on "
          f"{len(training_samples)} weighted proxy sample(s)…")
    train_loop(pipe.mula, training_samples)

    print("\nSaving outputs…")
    save_outputs(pipe.mula)

    subprocess.run(["gsutil", "-m", "rm", "-r", _GCS_RUN_BASE], check=False)
    print("\nPaired fine-tuning complete.")


if __name__ == "__main__":
    main()
