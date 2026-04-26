"""
generate_audio2audio.py — WavLM-conditioned audio generation.

Extends generate.py with true melodic conditioning: the input WAV is encoded
by a frozen WavLM encoder, projected to backbone dimension, and injected as
NUM_PREFIX_TOKENS conditioning vectors prepended to every backbone forward
pass.

If trained AudioConditioningModule weights exist in GCS (produced by
finetune_audio2audio.py) they are loaded automatically.  Without them the
script still runs but uses a random projection — melodic conditioning will be
weak.  Run finetune_audio2audio.py first for best results.

Environment variables (all from .env):
    GCS_BUCKET_NAME, GCS_BUCKET_FOLDER_PREFIX, MODEL_SIZE, RUN_MODE
    INPUT_WAV           GCS path or local path to reference input wav
    GCS_TAGS_FILE       GCS path to pre-existing tags .txt (optional)
    NUM_OUTPUTS         Number of clips to generate  (default: 3)
    MAX_AUDIO_MS        Max clip length in ms         (default: 30000)
    TEMPERATURE         Sampling temperature          (default: 1.0)
    CFG_SCALE           Classifier-free guidance      (default: 3.0)
    AUDIO_ENCODER_MODEL         WavLM HF model id             (default: microsoft/wavlm-base)
    NUM_PREFIX_TOKENS   Conditioning prefix length    (default: 32)
"""

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── annotation helper ──────────────────────────────────────────────────────────
def _try_annotate(local_wav: str) -> tuple:
    wav_path = Path(local_wav)
    try:
        app_dir = Path(__file__).resolve().parent.parent
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        from scripts.annotate import annotate_file
        return annotate_file(wav_path, data_dir=wav_path.parent)
    except Exception as exc:
        print(f"  [annotate] warning: {exc}")
        return None, None


# ── heartlib / peft ────────────────────────────────────────────────────────────
try:
    from heartlib import HeartMuLaGenPipeline
except ImportError as e:
    sys.exit(f"heartlib not available: {e}")

from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
GCS_BUCKET_NAME          = os.environ["GCS_BUCKET_NAME"]
GCS_BUCKET_FOLDER_PREFIX = os.getenv("GCS_BUCKET_FOLDER_PREFIX", "heartmula")
MODEL_SIZE               = os.getenv("MODEL_SIZE", "3b").upper()
_MODEL_SIZE_LOWER        = MODEL_SIZE.lower()
RUN_MODE                 = os.getenv("RUN_MODE", "train")
INPUT_WAV                = os.getenv("INPUT_WAV", "")
GCS_TAGS_FILE            = os.getenv("GCS_TAGS_FILE", "")
NUM_OUTPUTS              = int(os.getenv("NUM_OUTPUTS",        "3"))
MAX_AUDIO_MS             = int(os.getenv("MAX_AUDIO_MS",       "30000"))
TEMPERATURE              = float(os.getenv("TEMPERATURE",      "1.0"))
CFG_SCALE                = float(os.getenv("CFG_SCALE",        "3.0"))
AUDIO_ENCODER_MODEL              = os.getenv("AUDIO_ENCODER_MODEL",            "m-a-p/MERT-v1-95M")
NUM_PREFIX_TOKENS        = int(os.getenv("NUM_PREFIX_TOKENS",  "32"))

CKPT_DIR = Path("./ckpt")
OUT_DIR  = Path("./out/generated")

GCS_MODEL_CACHE  = f"gs://{GCS_BUCKET_NAME}/model-cache"
GCS_RUN_BASE     = f"gs://{GCS_BUCKET_NAME}/{GCS_BUCKET_FOLDER_PREFIX}-{_MODEL_SIZE_LOWER}/{RUN_MODE}"
GCS_ADAPTER_PATH = f"{GCS_RUN_BASE}/latest/adapter"
GCS_AUDIO_COND   = f"{GCS_RUN_BASE}/latest/audio_conditioning"
GCS_OUTPUT_PATH  = f"{GCS_RUN_BASE}/latest/generated"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

_ENCODER_DIM_MAP = {
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
ENCODER_DIM  = _ENCODER_DIM_MAP.get(AUDIO_ENCODER_MODEL, 768)
ENCODER_SR = _ENCODER_SR_MAP.get(AUDIO_ENCODER_MODEL, 24000)

print(f"\nDevice          : {DEVICE}")
print(f"Model           : HeartMuLa-oss-{MODEL_SIZE}  +  LoRA from {GCS_ADAPTER_PATH}")
print(f"Input           : {INPUT_WAV or '(none)'}")
print(f"Audio encoder   : {AUDIO_ENCODER_MODEL}  ({ENCODER_DIM}d)  →  {NUM_PREFIX_TOKENS} prefix tokens")
print(f"Output          : {NUM_OUTPUTS} clips  max={MAX_AUDIO_MS}ms  temp={TEMPERATURE}  cfg={CFG_SCALE}\n")


# ── AudioConditioningModule ────────────────────────────────────────────────────

class AudioConditioningModule(nn.Module):
    """
    WavLM frame features → NUM_PREFIX_TOKENS conditioning vectors.

    WavLM output is adaptively pooled to NUM_PREFIX_TOKENS time steps (preserving
    temporal/melodic ordering), projected to backbone_dim, and summed with
    learned positional embeddings.
    """

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


# ── Prefix conditioning via additive bias hook ────────────────────────────────
#
# HeartMuLa inference uses generate_frame → backbone(h, input_pos=..., mask=...)
# with KV-cache position tracking and RoPE.  Prepending tokens to the sequence
# changes its length while input_pos stays fixed, breaking RoPE.
#
# Instead: pool the P prefix tokens to a single global conditioning vector and
# add it to every hidden-state position at each backbone forward call.
# This carries the WavLM melodic information without touching sequence length,
# input_pos, RoPE, or KV cache.

def _make_prefix_hooks(prefix: torch.Tensor):
    """
    Return (pre_hook,) that adds global WavLM conditioning to backbone hidden states.

    prefix : (1, P, D) — pooled to (1, 1, D) and broadcast-added to h.
    """
    # Pool P prefix tokens → single global conditioning vector (1, 1, D)
    cond = prefix.mean(dim=1, keepdim=True).detach()  # (1, 1, D)

    def pre_hook(module, args):
        if not args or not isinstance(args[0], torch.Tensor) or args[0].dim() != 3:
            return args
        h = args[0]
        bias = cond.expand(h.shape[0], h.shape[1], -1).to(h.dtype)
        return (h + bias,) + args[1:]

    return (pre_hook,)


# ── helpers ────────────────────────────────────────────────────────────────────

def _gcs_dir_exists(gcs_path: str) -> bool:
    return subprocess.run(["gsutil", "ls", gcs_path], capture_output=True).returncode == 0


def _gcs_cp(src: str, dst: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", "-r", src, dst], check=True)


# ── 1. Download base model checkpoints ────────────────────────────────────────

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
    print(f"Loading audio encoder: {AUDIO_ENCODER_MODEL}…")
    wavlm = AutoModel.from_pretrained(AUDIO_ENCODER_MODEL, trust_remote_code=True)
    # Zero out relative-position weights that transformers leaves randomly initialized
    # when loading a mert_model via the WavLMModel fallback — these cause NaN features.
    for name, param in wavlm.named_parameters():
        if 'gru_rel_pos' in name or 'rel_attn_embed' in name:
            nn.init.zeros_(param)
    wavlm.eval()
    for p in wavlm.parameters():
        p.requires_grad_(False)
    return wavlm.to(DEVICE)


# ── 3. Load pipeline + LoRA adapter ───────────────────────────────────────────

def load_pipeline_with_adapter() -> HeartMuLaGenPipeline:
    print("Loading base HeartMuLa pipeline…")
    pipe = HeartMuLaGenPipeline.from_pretrained(
        pretrained_path=str(CKPT_DIR),
        device={"mula": DEVICE, "codec": "cpu"},
        dtype={"mula": DTYPE, "codec": torch.float32},
        version=MODEL_SIZE,
    )
    print(f"Downloading LoRA adapter from {GCS_ADAPTER_PATH}…")
    _gcs_cp(GCS_ADAPTER_PATH, str(CKPT_DIR) + "/")
    local_adapter = CKPT_DIR / Path(GCS_ADAPTER_PATH).name
    print("Applying LoRA adapter…")
    pipe.mula.backbone = PeftModel.from_pretrained(pipe.mula.backbone, str(local_adapter))
    pipe.mula.backbone = pipe.mula.backbone.merge_and_unload()
    print("Adapter merged.")
    return pipe


# ── 4. Build and load AudioConditioningModule ──────────────────────────────────

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


def load_conditioning_module(pipe: HeartMuLaGenPipeline) -> AudioConditioningModule:
    backbone_dim = _detect_backbone_dim(pipe)
    print(f"Backbone dim   : {backbone_dim}")

    module = AudioConditioningModule(ENCODER_DIM, backbone_dim, NUM_PREFIX_TOKENS).to(DEVICE)
    local_cond   = CKPT_DIR / "audio_conditioning"
    weights_file = local_cond / "module.pt"

    if _gcs_dir_exists(GCS_AUDIO_COND):
        print(f"Loading trained AudioConditioningModule from {GCS_AUDIO_COND}…")
        _gcs_cp(GCS_AUDIO_COND, str(CKPT_DIR) + "/")
        module.load_state_dict(torch.load(str(weights_file), map_location=DEVICE))
        print("  Conditioning module loaded.")
    else:
        print(
            f"  WARNING: No trained AudioConditioningModule found at {GCS_AUDIO_COND}.\n"
            "  Using random projection — melodic conditioning will be weak.\n"
            "  Run finetune_audio2audio.py first for best results."
        )

    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


# ── 5. Extract WavLM features ──────────────────────────────────────────────────

def extract_wavlm_features(wavlm, wav_path: str) -> torch.Tensor:
    import torchaudio
    print(f"Extracting WavLM features from {wav_path}…")
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != ENCODER_SR:
        waveform = torchaudio.functional.resample(waveform, sr, ENCODER_SR)
    waveform = waveform.to(DEVICE)  # (1, T)
    with torch.no_grad():
        out = wavlm(waveform)
    features = out.last_hidden_state  # (1, T_frames, encoder_dim)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Encoder: {features.shape[1]} frames from {waveform.shape[1] / ENCODER_SR:.1f}s audio")
    return features


# ── 6. Install prefix conditioning into backbone via hooks ────────────────────

def install_prefix_backbone(pipe: HeartMuLaGenPipeline, prefix: torch.Tensor) -> None:
    (pre_hook,) = _make_prefix_hooks(prefix)
    pipe.mula.backbone.register_forward_pre_hook(pre_hook)


# ── 7. Resolve input / tags ────────────────────────────────────────────────────

def resolve_tags_file():
    if not GCS_TAGS_FILE:
        return None
    local = Path("./data/paired/tags/input.txt")
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tags file from {GCS_TAGS_FILE}…")
    subprocess.run(["gsutil", "cp", GCS_TAGS_FILE, str(local)], check=True)
    print(f"  Tags : {local.read_text().strip()}")
    return local



def resolve_input_wav() -> str:
    if not INPUT_WAV:
        return ""
    if INPUT_WAV.startswith("gs://"):
        local = Path("./data/input.wav")
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading input wav from {INPUT_WAV}…")
        subprocess.run(["gsutil", "cp", INPUT_WAV, str(local)], check=True)
        return str(local)
    return INPUT_WAV


# ── 8. Generate + postprocess ──────────────────────────────────────────────────

def generate_clip(pipe, input_wav, idx, tags_path=None, lyrics_path=None) -> dict:
    print(f"  Generating clip {idx + 1}/{NUM_OUTPUTS}…", end=" ", flush=True)

    if tags_path is None or lyrics_path is None:
        tmp = Path("./data/tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        if tags_path is None:
            tags_path = tmp / "tags.txt"
            tags_path.write_text("")
        if lyrics_path is None:
            lyrics_path = tmp / "lyrics.txt"
            lyrics_path.write_text("")

    inputs = {"tags": str(tags_path), "lyrics": str(lyrics_path)}
    if input_wav:
        inputs["reference_audio"] = input_wav

    try:
        model_inputs = pipe.preprocess(inputs, cfg_scale=CFG_SCALE)
    except TypeError:
        model_inputs = pipe.preprocess(inputs)

    with torch.no_grad():
        try:
            model_outputs = pipe._forward(
                model_inputs,
                max_audio_length_ms=MAX_AUDIO_MS,
                temperature=TEMPERATURE,
                topk=50,
                cfg_scale=CFG_SCALE,
            )
        except TypeError:
            model_outputs = pipe._forward(model_inputs, max_audio_length_ms=MAX_AUDIO_MS)

    frames = model_outputs["frames"]
    print(f"{frames.shape[1]} frames  ({frames.shape[1] / 12.5:.1f} s)")
    return model_outputs


def postprocess_and_save(pipe, model_outputs: dict, idx: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"generated_{idx + 1:03d}.wav"
    pipe.postprocess(model_outputs, str(out_path))
    print(f"  Saved → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not INPUT_WAV:
        sys.exit("INPUT_WAV is required for audio-to-audio generation.")

    download_checkpoints()

    wavlm = load_wavlm()
    pipe  = load_pipeline_with_adapter()
    pipe.mula.eval()

    module    = load_conditioning_module(pipe)
    input_wav = resolve_input_wav()
    tags_path = resolve_tags_file()

    # Build WavLM prefix and install into backbone
    features = extract_wavlm_features(wavlm, input_wav)
    prefix   = module(features)  # (1, P, backbone_dim)
    print(f"Prefix shape   : {tuple(prefix.shape)}")

    del wavlm, features, module
    torch.cuda.empty_cache()

    install_prefix_backbone(pipe, prefix)
    print("Prefix conditioning installed.\n")

    # Annotate — always run Whisper; use GCS_TAGS_FILE tags if supplied, else CLAP
    if input_wav:
        print("Annotating input wav…")
        auto_tags, lyrics_path = _try_annotate(input_wav)
        if tags_path is None:
            tags_path = auto_tags

        if tags_path:
            print(f"  [annotate] tags   → {tags_path}")
        if lyrics_path:
            print(f"  [annotate] lyrics → {lyrics_path}")
            gcs_lyrics_dest = f"{GCS_OUTPUT_PATH}/lyrics_input.txt"
            subprocess.run(["gsutil", "cp", str(lyrics_path), gcs_lyrics_dest], check=False)
            print(f"  [annotate] lyrics uploaded → {gcs_lyrics_dest}")

    output_files = []
    for i in range(NUM_OUTPUTS):
        model_outputs = generate_clip(pipe, input_wav, i, tags_path, lyrics_path)
        out_path      = postprocess_and_save(pipe, model_outputs, i)
        output_files.append(out_path)
        print(f"  Uploading {out_path.name} to GCS…")
        _gcs_cp(str(out_path), GCS_OUTPUT_PATH + "/")
        torch.cuda.empty_cache()

    print(f"\nAll {len(output_files)} clip(s) uploaded to {GCS_OUTPUT_PATH}.")
    print("Done.")


if __name__ == "__main__":
    main()
