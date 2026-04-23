"""
finetune_audio2audio.py — train AudioConditioningModule + LoRA for WavLM-conditioned generation.

For each (input.wav, output.wav) pair:
  1. Encode input.wav with frozen WavLM → prefix tokens via AudioConditioningModule
  2. Prepend prefix to backbone hidden states (PrefixBackbone)
  3. Build teacher-forcing target from output.wav reference (same approach as finetune.py)
  4. Cross-entropy loss on codebook-0 at audio positions

Only the AudioConditioningModule projection + positional embeddings and the LoRA
adapter are trained.  WavLM stays frozen.

Run this before generate_audio2audio.py to get proper melodic conditioning.

Required env vars:
    GCS_BUCKET_NAME

Optional env vars:
    GCS_BUCKET_FOLDER_PREFIX   (default: "heartmula")
    MODEL_SIZE                 (default: "3b")
    RUN_MODE                   (default: "train")
    PAIRED_DATA_DIR            (default: "./data/paired")
    NUM_EPOCHS                 (default: 3)
    LEARNING_RATE              (default: 5e-5)
    LORA_RANK                  (default: 16)
    MAX_AUDIO_MS               (default: 30000)
    WAVLM_MODEL                (default: microsoft/wavlm-base)
    NUM_PREFIX_TOKENS          (default: 32)
"""

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "torchaudio"], check=True)
    import torchaudio

try:
    import heartlib  # noqa: F401
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps",
         "git+https://github.com/HeartMuLa/heartlib.git"], check=True)

from heartlib import HeartMuLaGenPipeline
from peft import LoraConfig, get_peft_model

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CODEBOOKS = 8
EMPTY_ID      = 0
CODEC_FPS     = 12.5

# ── Config ─────────────────────────────────────────────────────────────────────
GCS_BUCKET_NAME          = os.environ["GCS_BUCKET_NAME"]
GCS_BUCKET_FOLDER_PREFIX = os.getenv("GCS_BUCKET_FOLDER_PREFIX", "heartmula")
MODEL_SIZE               = os.getenv("MODEL_SIZE", "3b").upper()
_MODEL_SIZE_LOWER        = MODEL_SIZE.lower()
RUN_MODE                 = os.getenv("RUN_MODE", "train")
PAIRED_DATA_DIR          = Path(os.getenv("PAIRED_DATA_DIR", "./data/paired"))
NUM_EPOCHS               = int(os.getenv("NUM_EPOCHS",    "3"))
LEARNING_RATE            = float(os.getenv("LEARNING_RATE", "5e-5"))
LORA_RANK                = int(os.getenv("LORA_RANK",     "16"))
MAX_AUDIO_MS             = int(os.getenv("MAX_AUDIO_MS",  "30000"))
LORA_ALPHA               = LORA_RANK * 2
WAVLM_MODEL              = os.getenv("WAVLM_MODEL",           "m-a-p/MERT-v1-95M")
NUM_PREFIX_TOKENS        = int(os.getenv("NUM_PREFIX_TOKENS", "32"))

CKPT_DIR           = Path("./ckpt")
OUTPUT_ADAPTER_DIR = Path("./out/adapters/audio2audio")
OUTPUT_COND_DIR    = Path("./out/audio_conditioning")

GCS_MODEL_CACHE   = f"gs://{GCS_BUCKET_NAME}/model-cache"
_GCS_RUN_BASE     = f"gs://{GCS_BUCKET_NAME}/{GCS_BUCKET_FOLDER_PREFIX}-{_MODEL_SIZE_LOWER}/{RUN_MODE}"
GCS_OUTPUT_BASE   = f"{_GCS_RUN_BASE}/latest"
GCS_AUDIO_COND    = f"{GCS_OUTPUT_BASE}/audio_conditioning"
_GCS_CKPT_BASE    = f"{_GCS_RUN_BASE}/run-checkpoint-a2a"
SAMPLES_CKPT_GCS  = f"{_GCS_CKPT_BASE}/samples.pt"
LORA_CKPT_GCS     = f"{_GCS_CKPT_BASE}/lora_adapter"
COND_CKPT_GCS     = f"{_GCS_CKPT_BASE}/audio_conditioning"
EPOCH_CKPT_GCS    = f"{_GCS_CKPT_BASE}/epoch.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

_WAVLM_DIM_MAP = {
    "microsoft/wavlm-base":       768,
    "microsoft/wavlm-base-plus":  768,
    "microsoft/wavlm-large":     1024,
    "m-a-p/MERT-v1-95M":          768,
    "m-a-p/MERT-v1-330M":        1024,
}
_ENCODER_SR_MAP = {
    "microsoft/wavlm-base":      16000,
    "microsoft/wavlm-base-plus": 16000,
    "microsoft/wavlm-large":     16000,
    "m-a-p/MERT-v1-95M":         24000,
    "m-a-p/MERT-v1-330M":        24000,
}
WAVLM_DIM    = _WAVLM_DIM_MAP.get(WAVLM_MODEL, 768)
ENCODER_SR   = _ENCODER_SR_MAP.get(WAVLM_MODEL, 24000)

print(f"\nDevice          : {DEVICE}")
print(f"Model           : HeartMuLa-oss-{MODEL_SIZE}")
print(f"WavLM           : {WAVLM_MODEL}  ({WAVLM_DIM}d)  →  {NUM_PREFIX_TOKENS} prefix tokens")
print(f"Train           : audio2audio  {NUM_EPOCHS} epochs  lr={LEARNING_RATE}  LoRA rank={LORA_RANK}")
print(f"Data            : {PAIRED_DATA_DIR}\n")


# ── AudioConditioningModule ────────────────────────────────────────────────────

class AudioConditioningModule(nn.Module):
    def __init__(self, wavlm_dim: int, backbone_dim: int, num_prefix_tokens: int):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.pool = nn.AdaptiveAvgPool1d(num_prefix_tokens)
        self.proj = nn.Linear(wavlm_dim, backbone_dim)
        self.pos_emb = nn.Embedding(num_prefix_tokens, backbone_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # persistent=False: excluded from state_dict so existing checkpoints stay compatible
        self.register_buffer('_pos_idx', torch.arange(num_prefix_tokens), persistent=False)

    def forward(self, wavlm_features: torch.Tensor) -> torch.Tensor:
        # wavlm_features: (1, T, wavlm_dim)
        x = self.pool(wavlm_features.transpose(1, 2)).transpose(1, 2)  # (1, P, wavlm_dim)
        x = self.proj(x)                                                 # (1, P, backbone_dim)
        pos = self.pos_emb(self._pos_idx)
        return (x + pos.unsqueeze(0)).to(DTYPE)                          # (1, P, backbone_dim)


# ── PrefixBackbone wrapper ─────────────────────────────────────────────────────

class PrefixBackbone(nn.Module):
    """Wraps the backbone to prepend a mutable conditioning prefix."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.prefix = None  # set per-sample during training

    @property
    def _P(self) -> int:
        return self.prefix.shape[1] if self.prefix is not None else 0

    def set_prefix(self, prefix: torch.Tensor) -> None:
        self.prefix = prefix  # (1, P, D)

    def forward(self, h: torch.Tensor, mask=None, **kwargs):
        if self.prefix is None:
            return self.backbone(h, mask=mask, **kwargs)

        B, S, D = h.shape
        P = self._P
        prefix = self.prefix.expand(B, -1, -1).to(h.dtype)
        h_aug = torch.cat([prefix, h], dim=1)  # (B, P+S, D)

        if mask is not None and mask.dim() == 3:
            new_S = P + S
            new_mask = torch.zeros(B, new_S, new_S, dtype=mask.dtype, device=mask.device)
            new_mask[:, :P, :P] = True
            new_mask[:, P:, :P] = True
            new_mask[:, P:, P:] = mask
            mask = new_mask

        out = self.backbone(h_aug, mask=mask, **kwargs)

        if isinstance(out, torch.Tensor):
            return out[:, P:, :]
        main = out[0][:, P:, :]
        return (main,) + out[1:]


# ── helpers ────────────────────────────────────────────────────────────────────

def _gcs_dir_exists(gcs_path: str) -> bool:
    return subprocess.run(["gsutil", "ls", gcs_path], capture_output=True).returncode == 0


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


# ── 2. Load WavLM (frozen) ─────────────────────────────────────────────────────

def load_wavlm():
    from transformers import AutoModel
    print(f"Loading audio encoder: {WAVLM_MODEL} (on CPU to preserve GPU VRAM)…")
    wavlm = AutoModel.from_pretrained(WAVLM_MODEL, trust_remote_code=True)
    # Zero out relative-position weights that transformers leaves randomly initialized
    # when loading a mert_model via the WavLMModel fallback — these cause NaN features.
    for name, param in wavlm.named_parameters():
        if 'gru_rel_pos' in name or 'rel_attn_embed' in name:
            nn.init.zeros_(param)
    wavlm.eval()
    for p in wavlm.parameters():
        p.requires_grad_(False)
    return wavlm.cpu()  # kept on CPU; HeartMuLa needs all GPU VRAM


# ── 3. Load pipeline ───────────────────────────────────────────────────────────

def load_pipeline() -> HeartMuLaGenPipeline:
    print("Loading HeartMuLa pipeline…")
    return HeartMuLaGenPipeline.from_pretrained(
        pretrained_path=str(CKPT_DIR),
        device={"mula": DEVICE, "codec": DEVICE},
        dtype={"mula": DTYPE, "codec": DTYPE},
        version=MODEL_SIZE,
    )


# ── 4. Detect backbone dim ─────────────────────────────────────────────────────

def _detect_backbone_dim(pipe: HeartMuLaGenPipeline) -> int:
    bb = pipe.mula.backbone
    for attr in ("hidden_size", "d_model", "n_embd"):
        val = getattr(getattr(bb, "config", None), attr, None)
        if val:
            return val
    for p in bb.parameters():
        if p.dim() >= 2:
            return p.shape[-1]
    raise RuntimeError("Cannot detect backbone hidden dimension.")


# ── 5. Extract WavLM features (cached per file) ────────────────────────────────

def extract_wavlm_features(wavlm, wav_path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != ENCODER_SR:
        waveform = torchaudio.functional.resample(waveform, sr, ENCODER_SR)
    # Run encoder on CPU; features stored on CPU and moved to GPU only during training
    with torch.no_grad():
        out = wavlm(waveform.cpu())
    features = out.last_hidden_state.cpu()  # (1, T_frames, wavlm_dim)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ── 6. Discover audio pairs ────────────────────────────────────────────────────

def discover_audio_pairs() -> list:
    inputs_dir  = PAIRED_DATA_DIR / "inputs"
    outputs_dir = PAIRED_DATA_DIR / "outputs"
    lyrics_dir  = PAIRED_DATA_DIR / "lyrics"
    tags_dir    = PAIRED_DATA_DIR / "tags"
    if not inputs_dir.exists():
        raise FileNotFoundError(f"No inputs/ directory found under {PAIRED_DATA_DIR}.")
    pairs = []
    for wav in sorted(inputs_dir.glob("*.wav")):
        out = outputs_dir / wav.name
        if out.exists():
            lyrics_file = lyrics_dir / (wav.stem + ".txt")
            tags_file   = tags_dir   / (wav.stem + ".txt")
            pairs.append({
                "input":   wav,
                "output":  out,
                "lyrics":  lyrics_file if lyrics_file.exists() else None,
                "tags":    tags_file   if tags_file.exists()   else None,
            })
    if not pairs:
        raise FileNotFoundError(f"No matched .wav pairs found in {PAIRED_DATA_DIR}.")
    print(f"Found {len(pairs)} audio pair(s).")
    return pairs


# ── 7. Build training samples ──────────────────────────────────────────────────

def build_samples(pipe, wavlm, cond_module: AudioConditioningModule, pairs: list) -> list:
    """
    For each pair, encode input.wav with WavLM to get the conditioning prefix
    and derive teacher-forcing targets from output.wav (same proxy approach as finetune.py).
    """
    samples = []
    for i, pair in enumerate(pairs):
        print(f"  Pair {i + 1}/{len(pairs)}: {pair['input'].name} → {pair['output'].name}", flush=True)

        # Build conditioning prefix from input audio
        features = extract_wavlm_features(wavlm, pair["input"])
        with torch.no_grad():
            prefix = cond_module(features)  # (1, P, backbone_dim)

        tmp = Path("./data/tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        lyrics_path = pair["lyrics"] or (tmp / "lyrics.txt")
        tags_path   = pair["tags"]   or (tmp / "tags.txt")
        if not pair["lyrics"]:
            lyrics_path.write_text("")
        if not pair["tags"]:
            tags_path.write_text("")

        prompt_inputs = {
            "tags":            str(tags_path),
            "lyrics":          str(lyrics_path),
            "reference_audio": str(pair["output"]),
        }

        try:
            model_inputs = pipe.preprocess(prompt_inputs, cfg_scale=1.0)
        except TypeError:
            model_inputs = pipe.preprocess(prompt_inputs)

        print("    generating target codes from output reference…", end="  ", flush=True)
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

        audio_codes = model_outputs["frames"]  # (8, T)
        print(f"{audio_codes.shape[1]} frames ({audio_codes.shape[1] / CODEC_FPS:.1f} s)")

        samples.append({
            "prefix":        prefix.cpu(),
            "prompt_tokens": model_inputs["tokens"].cpu(),
            "audio_codes":   audio_codes.cpu(),
        })
    return samples


def load_or_build_samples(pipe, wavlm, cond_module) -> list:
    if _gcs_dir_exists(SAMPLES_CKPT_GCS):
        print("Resuming: loading samples from GCS checkpoint…")
        local = Path("./data/samples_a2a.pt")
        _gcs_download(SAMPLES_CKPT_GCS, str(local))
        raw = torch.load(str(local), map_location=DEVICE)
        return [
            {"prefix":        s["prefix"].to(DEVICE),
             "prompt_tokens": s["prompt_tokens"].to(DEVICE),
             "audio_codes":   s["audio_codes"].to(DEVICE)}
            for s in raw
        ]

    print("\nBuilding training samples from paired audio…")
    pairs   = discover_audio_pairs()
    samples = build_samples(pipe, wavlm, cond_module, pairs)

    local = Path("./data/samples_a2a.pt")
    local.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        [{"prefix":        s["prefix"].cpu(),
          "prompt_tokens": s["prompt_tokens"].cpu(),
          "audio_codes":   s["audio_codes"].cpu()}
         for s in samples],
        str(local),
    )
    _gcs_upload(str(local), SAMPLES_CKPT_GCS)
    print("Encoded samples cached to GCS.")
    return samples


# ── 8. Training sequence builder ──────────────────────────────────────────────

def build_training_sequence(prompt_tokens: torch.Tensor, audio_codes: torch.Tensor):
    if prompt_tokens.shape[0] == 2:
        prompt_tokens = prompt_tokens[:1]

    T_text  = prompt_tokens.shape[1]
    T_audio = audio_codes.shape[1]
    T_total = T_text + T_audio

    tokens = torch.full((1, T_total, NUM_CODEBOOKS + 1), EMPTY_ID, dtype=torch.long, device=DEVICE)
    mask   = torch.zeros((1, T_total, NUM_CODEBOOKS + 1), dtype=torch.bool, device=DEVICE)

    tokens[0, :T_text, :]              = prompt_tokens[0].to(DEVICE)
    mask[0, :T_text, -1]               = True

    tokens[0, T_text:, :NUM_CODEBOOKS] = audio_codes.T.long().to(DEVICE)
    mask[0, T_text:, :NUM_CODEBOOKS]   = True

    return tokens, mask, T_text


# ── 9. Setup LoRA + PrefixBackbone ────────────────────────────────────────────

def disable_backbone_caches(mula) -> None:
    for module in mula.backbone.modules():
        if hasattr(module, "kv_cache"):
            module.kv_cache = None
    if hasattr(mula.backbone, "_validate_inputs"):
        mula.backbone._validate_inputs = lambda *args, **kwargs: None


def setup_model(mula, backbone_dim: int) -> tuple:
    """Wrap backbone with PrefixBackbone, apply LoRA, return (prefix_backbone, cond_module)."""
    # Install mutable PrefixBackbone
    prefix_backbone = PrefixBackbone(mula.backbone)
    mula.backbone = prefix_backbone

    # LoRA on the inner backbone
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(prefix_backbone.backbone, lora_cfg)
    prefix_backbone.backbone = peft_model
    mula._peft_model = peft_model
    peft_model.print_trainable_parameters()

    cond_module = AudioConditioningModule(WAVLM_DIM, backbone_dim, NUM_PREFIX_TOKENS).to(DEVICE)
    return prefix_backbone, cond_module


# ── 10. Forward pass with prefix ──────────────────────────────────────────────

def _backbone_hidden_states(mula, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    embeds = mula._embed_tokens(tokens, uncond_mask=None)            # (1, S, 9, D)
    h = (embeds * mask.unsqueeze(-1)).sum(dim=2).to(DTYPE)           # (1, S, D)
    S = h.shape[1]
    causal_mask = torch.tril(
        torch.ones(S, S, dtype=torch.bool, device=DEVICE)
    ).unsqueeze(0)
    return mula.backbone(h, mask=causal_mask)                        # (1, S, D)


def compute_loss(mula, tokens: torch.Tensor, mask: torch.Tensor, T_input: int) -> torch.Tensor:
    h = _backbone_hidden_states(mula, tokens[:, :-1, :], mask[:, :-1, :])
    audio_h = h[:, T_input - 1:, :].to(DTYPE)
    logits  = mula.codebook0_head(audio_h)
    targets = tokens[:, T_input:, 0]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


# ── 11. Training loop ──────────────────────────────────────────────────────────

def _load_start_epoch() -> int:
    result = subprocess.run(
        ["gsutil", "cat", EPOCH_CKPT_GCS], capture_output=True, text=True)
    if result.returncode == 0:
        epoch = int(result.stdout.strip())
        print(f"Resuming: starting from epoch {epoch + 1}.")
        return epoch + 1
    return 0


def train_loop(mula, prefix_backbone: PrefixBackbone, cond_module: AudioConditioningModule,
               samples: list) -> None:
    for p in mula.codebook0_head.parameters():
        p.requires_grad_(True)

    trainable  = [p for p in mula.backbone.parameters() if p.requires_grad]
    trainable += list(mula.codebook0_head.parameters())
    trainable += list(cond_module.parameters())
    optimizer  = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

    start_epoch = _load_start_epoch()
    if start_epoch > 0:
        if _gcs_dir_exists(LORA_CKPT_GCS):
            print("Resuming: loading LoRA adapter from GCS…")
            local_lora = Path("./ckpt/lora_adapter_a2a")
            _gcs_download(LORA_CKPT_GCS, str(Path("./ckpt")) + "/")
            mula._peft_model.load_adapter(str(local_lora), adapter_name="default")
        if _gcs_dir_exists(COND_CKPT_GCS):
            print("Resuming: loading AudioConditioningModule from GCS…")
            local_cond = Path("./ckpt/audio_conditioning_ckpt/module.pt")
            _gcs_download(COND_CKPT_GCS, str(Path("./ckpt")) + "/")
            cond_module.load_state_dict(torch.load(str(local_cond), map_location=DEVICE))

    mula.train()
    cond_module.train()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for sample in samples:
            # Set the per-sample prefix
            prefix = sample["prefix"].to(DEVICE)
            with torch.no_grad():
                pass  # prefix was precomputed; for training we recompute through cond_module
            # Re-run cond_module in train mode to allow gradients
            prefix_train = cond_module(
                # We stored raw features → re-extract would be cleaner, but we stored the
                # already-projected prefix for memory efficiency. Use it directly here.
                # cond_module is trainable so we need a differentiable path:
                # We'll update: store wavlm_features in samples instead of prefix.
                sample["prefix"].to(DEVICE).detach()  # no-op placeholder; see note below
            )
            # NOTE: To enable gradients through cond_module we stored wavlm_features, not prefix.
            # The line above is a placeholder. See _build_differentiable_prefix below.
            prefix_backbone.set_prefix(prefix_train if hasattr(sample, "wavlm_features") else prefix)

            tokens, mask, T_input = build_training_sequence(
                sample["prompt_tokens"], sample["audio_codes"]
            )
            loss = compute_loss(mula, tokens, mask, T_input)
            (loss / len(samples)).backward()
            epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS}  loss={epoch_loss / len(samples):.4f}")

        # Checkpoint LoRA
        local_lora = Path("./ckpt/lora_ckpt_a2a")
        mula._peft_model.save_pretrained(str(local_lora))
        _gcs_upload(str(local_lora), LORA_CKPT_GCS)
        # Checkpoint conditioning module
        local_cond_dir = Path("./ckpt/audio_conditioning_ckpt")
        local_cond_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cond_module.state_dict(), local_cond_dir / "module.pt")
        _gcs_upload(str(local_cond_dir), COND_CKPT_GCS)
        # Checkpoint epoch
        subprocess.run(
            ["gsutil", "cp", "-", EPOCH_CKPT_GCS],
            input=str(epoch), text=True, check=True)
        print(f"  Checkpoint saved after epoch {epoch + 1}.")

    mula.eval()
    cond_module.eval()


def build_samples_with_features(pipe, wavlm_features_cache: dict, pairs: list) -> list:
    """
    Build training samples reusing pre-computed WavLM features (no WavLM on GPU).
    Stores raw wavlm_features so cond_module gradients flow during training.
    """
    samples = []
    for i, pair in enumerate(pairs):
        print(f"  Pair {i + 1}/{len(pairs)}: {pair['input'].name} → {pair['output'].name}", flush=True)

        wavlm_features = wavlm_features_cache[str(pair["input"])]  # already on CPU

        tmp = Path("./data/tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        lyrics_path = pair["lyrics"] or (tmp / "lyrics.txt")
        tags_path   = pair["tags"]   or (tmp / "tags.txt")
        if not pair["lyrics"]:
            lyrics_path.write_text("")
        if not pair["tags"]:
            tags_path.write_text("")

        try:
            model_inputs = pipe.preprocess(
                {"tags": str(tags_path), "lyrics": str(lyrics_path),
                 "reference_audio": str(pair["output"])},
                cfg_scale=1.0)
        except TypeError:
            model_inputs = pipe.preprocess(
                {"tags": str(tags_path), "lyrics": str(lyrics_path),
                 "reference_audio": str(pair["output"])})

        print("    generating target codes…", end="  ", flush=True)
        with torch.no_grad():
            try:
                model_outputs = pipe._forward(
                    model_inputs, max_audio_length_ms=MAX_AUDIO_MS,
                    temperature=1.0, topk=50, cfg_scale=1.0)
            except TypeError:
                model_outputs = pipe._forward(model_inputs, max_audio_length_ms=MAX_AUDIO_MS)

        audio_codes = model_outputs["frames"]
        print(f"{audio_codes.shape[1]} frames ({audio_codes.shape[1] / CODEC_FPS:.1f} s)")

        samples.append({
            "wavlm_features": wavlm_features.cpu(),
            "prompt_tokens":  model_inputs["tokens"].cpu(),
            "audio_codes":    audio_codes.cpu(),
        })
    return samples


def train_loop_v2(mula, prefix_backbone: PrefixBackbone, cond_module: AudioConditioningModule,
                  samples: list) -> None:
    """Training loop that keeps wavlm_features and allows gradients through cond_module."""
    for p in mula.codebook0_head.parameters():
        p.requires_grad_(True)

    trainable  = [p for p in mula.backbone.parameters() if p.requires_grad]
    trainable += list(mula.codebook0_head.parameters())
    trainable += list(cond_module.parameters())
    optimizer  = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

    start_epoch = _load_start_epoch()
    if start_epoch > 0:
        if _gcs_dir_exists(LORA_CKPT_GCS):
            print("Resuming: loading LoRA adapter from GCS…")
            local_lora = Path("./ckpt") / "lora_adapter_a2a"
            _gcs_download(LORA_CKPT_GCS, str(Path("./ckpt")) + "/")
            mula._peft_model.load_adapter(str(local_lora), adapter_name="default")
        if _gcs_dir_exists(COND_CKPT_GCS):
            print("Resuming: loading AudioConditioningModule from GCS…")
            _gcs_download(COND_CKPT_GCS, str(Path("./ckpt")) + "/")
            cond_module.load_state_dict(
                torch.load(str(Path("./ckpt/audio_conditioning_ckpt/module.pt")),
                           map_location=DEVICE))

    mula.train()
    cond_module.train()

    # Pre-compute training sequences once — they are identical across all epochs
    precomputed_seqs = [
        build_training_sequence(s["prompt_tokens"], s["audio_codes"])
        for s in samples
    ]

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for sample, (tokens, mask, T_input) in zip(samples, precomputed_seqs):
            features = sample["wavlm_features"].to(DEVICE)
            prefix   = cond_module(features)  # differentiable: (1, P, D)
            prefix_backbone.set_prefix(prefix)

            loss = compute_loss(mula, tokens, mask, T_input)
            (loss / len(samples)).backward()
            epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS}  loss={epoch_loss / len(samples):.4f}")

        local_lora = Path("./ckpt/lora_ckpt_a2a")
        mula._peft_model.save_pretrained(str(local_lora))
        _gcs_upload(str(local_lora), LORA_CKPT_GCS)

        local_cond_dir = Path("./ckpt/audio_conditioning_ckpt")
        local_cond_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cond_module.state_dict(), local_cond_dir / "module.pt")
        _gcs_upload(str(local_cond_dir), COND_CKPT_GCS)

        subprocess.run(
            ["gsutil", "cp", "-", EPOCH_CKPT_GCS],
            input=str(epoch), text=True, check=True)
        print(f"  Checkpoint saved after epoch {epoch + 1}.")

    mula.eval()
    cond_module.eval()


# ── 12. Save outputs ───────────────────────────────────────────────────────────

def save_outputs(mula, cond_module: AudioConditioningModule) -> None:
    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_COND_DIR.mkdir(parents=True, exist_ok=True)

    mula._peft_model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    torch.save(mula.codebook0_head.state_dict(), OUTPUT_ADAPTER_DIR / "codebook0_head.pt")
    print(f"LoRA adapter          → {OUTPUT_ADAPTER_DIR}")

    torch.save(cond_module.state_dict(), OUTPUT_COND_DIR / "module.pt")
    print(f"AudioConditioningModule → {OUTPUT_COND_DIR}")

    # Upload conditioning module to the path generate_audio2audio.py expects
    _gcs_upload(str(OUTPUT_COND_DIR), GCS_AUDIO_COND)
    print(f"Uploaded conditioning  → {GCS_AUDIO_COND}")

    # Upload LoRA adapter to latest/adapter so generate_audio2audio.py can load it
    gcs_adapter = f"{GCS_OUTPUT_BASE}/adapter"
    _gcs_upload(str(OUTPUT_ADAPTER_DIR), gcs_adapter)
    print(f"Uploaded LoRA adapter  → {gcs_adapter}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    download_checkpoints()

    # Phase 1: encode all input WAVs with WavLM on CPU before loading HeartMuLa
    wavlm = load_wavlm()
    print("\nPre-encoding all input WAVs with WavLM (CPU)…")
    pairs = discover_audio_pairs()
    wavlm_features_cache = {}
    for pair in pairs:
        print(f"  WavLM encode: {pair['input'].name}")
        wavlm_features_cache[str(pair["input"])] = extract_wavlm_features(wavlm, pair["input"])
    del wavlm
    torch.cuda.empty_cache()
    print("WavLM encoding done; VRAM freed.\n")

    # Phase 2: load HeartMuLa (needs full GPU VRAM)
    pipe = load_pipeline()

    backbone_dim = _detect_backbone_dim(pipe)
    print(f"Backbone dim    : {backbone_dim}")

    # Build samples reusing cached features (no WavLM on GPU needed)
    print("\nBuilding training samples from paired audio…")
    samples = build_samples_with_features(pipe, wavlm_features_cache, pairs)
    torch.cuda.empty_cache()

    print("\nSetting up LoRA + PrefixBackbone…")
    disable_backbone_caches(pipe.mula)
    prefix_backbone, cond_module = setup_model(pipe.mula, backbone_dim)

    print(f"\nFine-tuning for {NUM_EPOCHS} epochs…")
    train_loop_v2(pipe.mula, prefix_backbone, cond_module, samples)

    print("\nSaving outputs…")
    save_outputs(pipe.mula, cond_module)

    subprocess.run(["gsutil", "-m", "rm", "-r", _GCS_CKPT_BASE], check=False)
    print("\nAudio-to-audio fine-tuning complete.")


if __name__ == "__main__":
    main()
