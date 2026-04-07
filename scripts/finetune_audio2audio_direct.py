#!/usr/bin/env python3
"""
finetune_audio2audio_direct.py — fast-shot direct supervised fine-tuning.

Problem with the proxy approach in finetune_audio2audio.py
-----------------------------------------------------------
HeartCodec has no public encoder, so we can't get ground-truth codec tokens
for the output WAV. Instead, we generate proxy samples and score them by
mel-similarity. This is slow (proxy generation per pair), noisy (proxy ≠ truth),
and ignores the output WAV as a direct training signal.

Direct loss approach
--------------------
WavLM encodes audio into rich semantic frame-level features. We use it to
supervise the conditioning projection directly:

    input_prefix  = projection(pool(WavLM(input_wav))) + pos_embed   ← trainable
    target_prefix = projection(pool(WavLM(output_wav))) + pos_embed   ← detached

    Loss = MSE(input_prefix, target_prefix)

This teaches the projection to map input audio into the same backbone-space
representation that the output audio would produce. During inference, the
backbone will then generate audio guided toward the output transformation.

Because no proxy generation or backbone forward pass is needed, the full
training loop runs in minutes (even on CPU) and converges in 50–200 steps.

Phase 2 (optional, backbone_align=True)
----------------------------------------
Optionally align in backbone hidden-state space instead of prefix space,
by doing a no_grad backbone forward for the target prefix and a gradient
backbone forward for the input prefix. This is stronger but slower.

Workflow
--------
1. Downloads existing adapter from GCS (from the prior audio2audio training).
2. Fine-tunes only AudioConditioningModule.projection + pos_embed.
3. Saves the updated adapter back to GCS.

Environment variables
---------------------
  GCS_BUCKET_NAME                (required)
  GCS_BUCKET_FOLDER_PREFIX       (default: heartmula)
  MODEL_SIZE                     (default: 3b)
  RUN_MODE                       (default: audio2audio)
  PAIRS_DIR                      (default: ./data/paired)
  STYLE_TAGS                     (default: piano,ambient,reflective)
  NUM_PREFIX_TOKENS              (default: 32)
  WAVLM_MODEL                    (default: microsoft/wavlm-base)
  NUM_STEPS                      Gradient steps per pair  (default: 100)
  LEARNING_RATE                  (default: 1e-4)
  BACKBONE_ALIGN                 Run phase-2 backbone alignment (default: false)
  BACKBONE_ALIGN_STEPS           Steps for phase 2         (default: 20)
"""

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
from peft import PeftModel  # noqa: E402
from transformers import WavLMModel, Wav2Vec2FeatureExtractor  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
WAVLM_SR  = 16000
WAVLM_DIM = 768    # wavlm-base; wavlm-large → 1024

# ── Config ─────────────────────────────────────────────────────────────────────
GCS_BUCKET_NAME          = os.environ["GCS_BUCKET_NAME"]
GCS_BUCKET_FOLDER_PREFIX = os.getenv("GCS_BUCKET_FOLDER_PREFIX", "heartmula")
MODEL_SIZE               = os.getenv("MODEL_SIZE", "3b").upper()
RUN_MODE                 = os.getenv("RUN_MODE", "audio2audio")
PAIRS_DIR                = Path(os.getenv("PAIRS_DIR", "./data/paired"))
STYLE_TAGS               = os.getenv("STYLE_TAGS", "piano,ambient,reflective")
NUM_PREFIX_TOKENS        = int(os.getenv("NUM_PREFIX_TOKENS", "32"))
WAVLM_MODEL              = os.getenv("WAVLM_MODEL",           "microsoft/wavlm-base")
NUM_STEPS                = int(os.getenv("NUM_STEPS",         "100"))
LEARNING_RATE            = float(os.getenv("LEARNING_RATE",   "1e-4"))
BACKBONE_ALIGN           = os.getenv("BACKBONE_ALIGN", "false").lower() == "true"
BACKBONE_ALIGN_STEPS     = int(os.getenv("BACKBONE_ALIGN_STEPS", "20"))

CKPT_DIR = Path("./ckpt")
TMP_DIR  = Path("./data/tmp")

GCS_MODEL_CACHE  = f"gs://{GCS_BUCKET_NAME}/model-cache"
GCS_RUN_BASE     = (f"gs://{GCS_BUCKET_NAME}/{GCS_BUCKET_FOLDER_PREFIX}"
                    f"-{MODEL_SIZE}/{RUN_MODE}")
GCS_ADAPTER_PATH = f"{GCS_RUN_BASE}/latest/adapter"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

print(f"\nDevice  : {DEVICE}")
print(f"Model   : HeartMuLa-oss-{MODEL_SIZE}")
print(f"WavLM   : {WAVLM_MODEL}")
print(f"Steps   : {NUM_STEPS}  lr={LEARNING_RATE}  backbone_align={BACKBONE_ALIGN}\n")


# ── Utilities ──────────────────────────────────────────────────────────────────

def _gcs_dir_exists(gcs_path: str) -> bool:
    return subprocess.run(["gsutil", "ls", gcs_path], capture_output=True).returncode == 0


def _gcs_cp(src: str, dst: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", "-r", src, dst], check=True)


# ── AudioConditioningModule (must match finetune_audio2audio.py) ───────────────

class AudioConditioningModule(nn.Module):
    def __init__(
        self,
        backbone_dim: int,
        num_prefix_tokens: int = NUM_PREFIX_TOKENS,
        wavlm_model_id: str = WAVLM_MODEL,
        encoder_dim: int = WAVLM_DIM,
    ) -> None:
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.backbone_dim      = backbone_dim

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_id)
        self.encoder = WavLMModel.from_pretrained(wavlm_model_id)
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        self.pos_embed  = nn.Parameter(torch.zeros(1, num_prefix_tokens, backbone_dim))
        self.projection = nn.Linear(encoder_dim, backbone_dim, bias=True)

    def _wavlm_features(self, wav_path: Path) -> torch.Tensor:
        """Returns pooled WavLM features (1, num_prefix_tokens, encoder_dim) on CPU."""
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != WAVLM_SR:
            waveform = torchaudio.functional.resample(waveform, sr, WAVLM_SR)

        with torch.no_grad():
            inputs = self.feature_extractor(
                waveform.squeeze(0).cpu().numpy(),
                sampling_rate=WAVLM_SR,
                return_tensors="pt",
            )
            features = self.encoder.cpu()(inputs["input_values"].cpu()).last_hidden_state
            features = F.adaptive_avg_pool1d(
                features.transpose(1, 2), self.num_prefix_tokens
            ).transpose(1, 2)  # (1, N, encoder_dim)

        return features  # CPU, no grad

    def encode_with_grad(self, wav_path: Path) -> torch.Tensor:
        """Encode WAV → prefix, gradient flows through projection + pos_embed."""
        features = self._wavlm_features(wav_path)
        prefix = self.projection(features.to(DEVICE, DTYPE))
        return prefix + self.pos_embed.to(DTYPE)  # (1, N, backbone_dim)

    def encode_no_grad(self, wav_path: Path) -> torch.Tensor:
        """Encode WAV → prefix with no gradient (used for target)."""
        with torch.no_grad():
            return self.encode_with_grad(wav_path).detach()


# ── 1. Download checkpoints ────────────────────────────────────────────────────

def download_checkpoints() -> Path:
    """Download base model (if needed) and existing adapter from GCS."""
    mula_dir  = CKPT_DIR / f"HeartMuLa-oss-{MODEL_SIZE}"
    codec_dir = CKPT_DIR / "HeartCodec-oss"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not (CKPT_DIR / "tokenizer.json").exists():
        gcs_src = f"{GCS_MODEL_CACHE}/HeartMuLaGen"
        if _gcs_dir_exists(gcs_src):
            _gcs_cp(f"{gcs_src}/*", str(CKPT_DIR) + "/")
        else:
            subprocess.run(
                ["huggingface-cli", "download", "--local-dir", str(CKPT_DIR),
                 "HeartMuLa/HeartMuLaGen"], check=True)

    # Backbone + codec are only needed for phase-2 backbone alignment.
    if BACKBONE_ALIGN:
        if not mula_dir.exists():
            gcs_src = f"{GCS_MODEL_CACHE}/HeartMuLa-oss-{MODEL_SIZE}"
            if _gcs_dir_exists(gcs_src):
                _gcs_cp(gcs_src, str(CKPT_DIR) + "/")
            else:
                subprocess.run(
                    ["huggingface-cli", "download", "--local-dir", str(mula_dir),
                     "HeartMuLa/HeartMuLa-oss-3B-happy-new-year"], check=True)

        if not codec_dir.exists():
            gcs_src = f"{GCS_MODEL_CACHE}/HeartCodec-oss"
            if _gcs_dir_exists(gcs_src):
                _gcs_cp(gcs_src, str(CKPT_DIR) + "/")
            else:
                subprocess.run(
                    ["huggingface-cli", "download", "--local-dir", str(codec_dir),
                     "HeartMuLa/HeartCodec-oss-20260123"], check=True)

    # Download existing adapter
    adapter_dir = CKPT_DIR / "adapter"
    if not adapter_dir.exists():
        if _gcs_dir_exists(GCS_ADAPTER_PATH):
            print(f"Downloading adapter from {GCS_ADAPTER_PATH}…")
            _gcs_cp(GCS_ADAPTER_PATH, str(CKPT_DIR) + "/")
        else:
            raise FileNotFoundError(
                f"No adapter found at {GCS_ADAPTER_PATH}. "
                "Run finetune_audio2audio.py first."
            )
    return adapter_dir


# ── 2. Load conditioning module from saved adapter ────────────────────────────

def load_conditioning_module(adapter_dir: Path) -> AudioConditioningModule:
    config_path  = adapter_dir / "audio_conditioning_config.pt"
    weights_path = adapter_dir / "audio_conditioning.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"audio_conditioning_config.pt not found in {adapter_dir}")

    config = torch.load(str(config_path), map_location="cpu")
    print(f"AudioConditioningModule config: {config}")

    module = AudioConditioningModule(
        backbone_dim=config["backbone_dim"],
        num_prefix_tokens=config["num_prefix_tokens"],
        wavlm_model_id=config["wavlm_model_id"],
    )
    module.projection = module.projection.to(DEVICE, dtype=DTYPE)
    module.pos_embed  = nn.Parameter(module.pos_embed.to(DEVICE, dtype=DTYPE))

    if weights_path.exists():
        state = torch.load(str(weights_path), map_location=DEVICE)
        module.load_state_dict(state)
        print("Loaded AudioConditioningModule weights.")

    return module


# ── 3. Load paired data ────────────────────────────────────────────────────────

def load_pairs() -> list[dict]:
    inputs_dir  = PAIRS_DIR / "inputs"
    outputs_dir = PAIRS_DIR / "outputs"
    tags_dir    = Path(os.getenv("TAGS_DIR", "./data/paired/tags"))

    pairs = []
    for input_path in sorted(inputs_dir.glob("*.wav")):
        s = input_path.stem
        output_path = outputs_dir / f"{s}.wav"
        if not output_path.exists():
            print(f"  [skip] {s}: no matching output WAV")
            continue
        tags_file = tags_dir / f"{s}.txt"
        tags = tags_file.read_text().strip() if tags_file.exists() else STYLE_TAGS
        pairs.append({"stem": s, "input": input_path, "output": output_path, "tags": tags})

    if not pairs:
        raise RuntimeError(f"No complete input/output pairs in {PAIRS_DIR}")

    print(f"Found {len(pairs)} pair(s):")
    for p in pairs:
        print(f"  {p['stem']}  tags='{p['tags']}'")
    return pairs


# ── 4. Phase 1: prefix alignment loss ────────────────────────────────────────
#
#  MSE(input_prefix, target_prefix.detach())
#
#  Teaches the projection: WavLM(input) → backbone_space(output)
#  Gradient only flows through the input path (target is detached).
#  No backbone forward needed. Runs in minutes.

def phase1_prefix_alignment(
    conditioning_module: AudioConditioningModule,
    pairs: list[dict],
) -> None:
    trainable = (
        list(conditioning_module.projection.parameters())
        + [conditioning_module.pos_embed]
    )
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

    # Pre-compute target prefixes (WavLM + current projection, then detach permanently)
    print("\n── Phase 1: prefix alignment ──────────────────────────────────────")
    print("Pre-computing target prefixes from output WAVs…")
    target_prefixes = []
    for p in pairs:
        tp = conditioning_module.encode_no_grad(p["output"])
        target_prefixes.append(tp)
        print(f"  {p['stem']} output prefix: {tuple(tp.shape)}")

    print(f"\nTraining projection for {NUM_STEPS} steps (lr={LEARNING_RATE})…")
    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        total_loss = 0.0

        for p, target_prefix in zip(pairs, target_prefixes):
            input_prefix = conditioning_module.encode_with_grad(p["input"])
            loss = F.mse_loss(input_prefix, target_prefix)
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        if (step + 1) % 10 == 0 or step == 0:
            avg = total_loss / len(pairs)
            print(f"  step {step + 1:>4}/{NUM_STEPS}  loss={avg:.6f}")

    print("Phase 1 complete.")


# ── 5. Phase 2 (optional): backbone hidden-state alignment ────────────────────
#
#  Stronger supervision: align the backbone's output hidden states, not just
#  the prefix. Requires a backbone forward per step — slower, but ensures the
#  backbone actually generates similar representations for input vs output.
#
#  teacher: backbone([target_prefix | dummy_text]) → h_target  (no_grad)
#  student: backbone([input_prefix  | dummy_text]) → h_student (gradient)
#  Loss = MSE(h_student[:, :N], h_target[:, :N])

def _dummy_text_tokens(pipe: HeartMuLaGenPipeline, tags: str) -> torch.Tensor:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tags_file = TMP_DIR / "direct_tags.txt"
    tags_file.write_text(tags.lower())
    (TMP_DIR / "direct_lyrics.txt").write_text("")
    try:
        model_inputs = pipe.preprocess(
            {"tags": str(tags_file), "lyrics": str(TMP_DIR / "direct_lyrics.txt")},
            cfg_scale=1.0,
        )
    except TypeError:
        model_inputs = pipe.preprocess(
            {"tags": str(tags_file), "lyrics": str(TMP_DIR / "direct_lyrics.txt")}
        )
    tokens = model_inputs["tokens"]
    if tokens.shape[0] == 2:
        tokens = tokens[:1]
    return tokens.to(DEVICE)  # (1, T_text, 9)


def phase2_backbone_alignment(
    conditioning_module: AudioConditioningModule,
    pipe: HeartMuLaGenPipeline,
    pairs: list[dict],
) -> None:
    print("\n── Phase 2: backbone hidden-state alignment ───────────────────────")

    NUM_CODEBOOKS = 8
    trainable = (
        list(conditioning_module.projection.parameters())
        + [conditioning_module.pos_embed]
    )
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE * 0.1)

    # Disable KV caches for prefix injection
    for module in pipe.mula.backbone.modules():
        if hasattr(module, "kv_cache"):
            module.kv_cache = None
    if hasattr(pipe.mula.backbone, "_validate_inputs"):
        pipe.mula.backbone._validate_inputs = lambda *a, **kw: None

    pipe.mula.eval()
    for p_param in pipe.mula.backbone.parameters():
        p_param.requires_grad_(False)

    # Build dummy text embeddings for each pair
    text_embeds_list = []
    for p in pairs:
        tokens = _dummy_text_tokens(pipe, p["tags"])
        mask = torch.zeros(*tokens.shape, dtype=torch.bool, device=DEVICE)
        mask[:, :, -1] = True
        embeds = pipe.mula._embed_tokens(tokens, uncond_mask=None)
        h_text = (embeds * mask.unsqueeze(-1)).sum(dim=2).to(DTYPE)
        text_embeds_list.append(h_text.detach())  # (1, T_text, D)

    N = conditioning_module.num_prefix_tokens

    def _backbone_prefix_hidden(prefix: torch.Tensor, h_text: torch.Tensor) -> torch.Tensor:
        h = torch.cat([prefix, h_text], dim=1)  # (1, N+T, D)
        S = h.shape[1]
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=DEVICE)).unsqueeze(0)
        return pipe.mula.backbone(h, mask=mask)[:, :N]  # (1, N, D) — prefix positions only

    print(f"Training for {BACKBONE_ALIGN_STEPS} steps…")
    for step in range(BACKBONE_ALIGN_STEPS):
        optimizer.zero_grad()
        total_loss = 0.0

        for p, h_text in zip(pairs, text_embeds_list):
            target_prefix = conditioning_module.encode_no_grad(p["output"])
            with torch.no_grad():
                h_target = _backbone_prefix_hidden(target_prefix, h_text)

            input_prefix = conditioning_module.encode_with_grad(p["input"])
            h_student = _backbone_prefix_hidden(input_prefix, h_text)

            loss = F.mse_loss(h_student, h_target)
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        avg = total_loss / len(pairs)
        print(f"  step {step + 1:>3}/{BACKBONE_ALIGN_STEPS}  loss={avg:.6f}")

    print("Phase 2 complete.")


# ── 6. Save updated adapter ───────────────────────────────────────────────────

def save_adapter(conditioning_module: AudioConditioningModule, adapter_dir: Path) -> None:
    """Overwrite conditioning weights in adapter_dir, keep LoRA files unchanged."""
    torch.save(conditioning_module.state_dict(), adapter_dir / "audio_conditioning.pt")
    print(f"Updated audio_conditioning.pt in {adapter_dir}")


def upload_adapter(adapter_dir: Path) -> None:
    import datetime
    ts = datetime.datetime.utcnow().strftime("%y-%m-%d-%H-%M-%S")
    gcs_ts = f"{GCS_RUN_BASE}/{ts}"
    gcs_latest = f"{GCS_RUN_BASE}/latest"

    print(f"Uploading to {gcs_ts}/adapter/ …")
    _gcs_cp(str(adapter_dir), f"{gcs_ts}/adapter")

    print(f"Updating {gcs_latest}/adapter/ …")
    subprocess.run(["gsutil", "-m", "rm", "-rf", f"{gcs_latest}/adapter"], check=False)
    _gcs_cp(str(adapter_dir), f"{gcs_latest}/adapter")
    print("Upload complete.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    adapter_dir = download_checkpoints()
    conditioning_module = load_conditioning_module(adapter_dir)
    pairs = load_pairs()

    # Phase 1: fast prefix alignment (always runs)
    phase1_prefix_alignment(conditioning_module, pairs)

    # Phase 2: backbone hidden-state alignment (optional, slower)
    if BACKBONE_ALIGN:
        print("\nLoading backbone for phase-2 alignment…")
        pipe = HeartMuLaGenPipeline.from_pretrained(
            pretrained_path=str(CKPT_DIR),
            device={"mula": DEVICE, "codec": DEVICE},
            dtype={"mula": DTYPE, "codec": DTYPE},
            version=MODEL_SIZE,
        )
        pipe.mula.backbone = PeftModel.from_pretrained(
            pipe.mula.backbone, str(adapter_dir)
        ).merge_and_unload()
        phase2_backbone_alignment(conditioning_module, pipe, pairs)

    save_adapter(conditioning_module, adapter_dir)
    upload_adapter(adapter_dir)
    print("\nDirect fine-tuning complete.")


if __name__ == "__main__":
    main()
