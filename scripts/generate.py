"""
generate.py — run inference with the fine-tuned HeartMuLa adapter.

Loads the base model, applies the LoRA adapter, and generates audio
conditioned on a reference input wav (no style tags needed).

Environment variables:
    GCS_BUCKET_NAME, GCS_BUCKET_FOLDER_PREFIX, MODEL_SIZE, RUN_MODE
    INPUT_WAV       GCS path or local path to the reference input wav
    GCS_TAGS_FILE   GCS path to a pre-existing tags .txt file (skips CLAP auto-annotation)
    NUM_OUTPUTS     Number of clips to generate  (default: 3)
    MAX_AUDIO_MS    Max clip length in ms         (default: 30000)
    TEMPERATURE     Sampling temperature          (default: 1.0)
    CFG_SCALE       Classifier-free guidance      (default: 3.0)
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

# ── annotation (optional — only imported when needed) ─────────────────────────
def _try_annotate(local_wav: str) -> tuple[Path | None, Path | None]:
    """
    Run annotate.py on *local_wav* to generate tags and lyrics.
    Returns (tags_path, lyrics_path) or (None, None) on failure.
    """
    wav_path = Path(local_wav)
    try:
        import sys
        app_dir = Path(__file__).resolve().parent.parent
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        from scripts.annotate import annotate_file
        tags_path, lyrics_path = annotate_file(wav_path, data_dir=wav_path.parent)
        return tags_path, lyrics_path
    except Exception as exc:
        print(f"  [annotate] warning: could not auto-annotate: {exc}")
        return None, None

# ── heartlib import (GPU-only) ─────────────────────────────────────────────────
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
INPUT_WAV                = os.getenv("INPUT_WAV", "")   # GCS path or local path
GCS_TAGS_FILE            = os.getenv("GCS_TAGS_FILE", "")    # optional GCS path to tags .txt
NUM_OUTPUTS              = int(os.getenv("NUM_OUTPUTS",   "3"))
MAX_AUDIO_MS             = int(os.getenv("MAX_AUDIO_MS",  "30000"))
TEMPERATURE              = float(os.getenv("TEMPERATURE", "1.0"))
CFG_SCALE                = float(os.getenv("CFG_SCALE",   "3.0"))

CKPT_DIR = Path("./ckpt")
OUT_DIR  = Path("./out/generated")

GCS_MODEL_CACHE  = f"gs://{GCS_BUCKET_NAME}/model-cache"
GCS_RUN_BASE     = f"gs://{GCS_BUCKET_NAME}/{GCS_BUCKET_FOLDER_PREFIX}-{_MODEL_SIZE_LOWER}/{RUN_MODE}"
GCS_ADAPTER_PATH = f"{GCS_RUN_BASE}/latest/adapter"
GCS_OUTPUT_PATH  = f"{GCS_RUN_BASE}/latest/generated"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

print(f"\nDevice : {DEVICE}")
print(f"Model  : HeartMuLa-oss-{MODEL_SIZE}  +  LoRA adapter from {GCS_ADAPTER_PATH}")
print(f"Input  : {INPUT_WAV or '(none — unconditioned)'}")
print(f"Output : {NUM_OUTPUTS} clips  max={MAX_AUDIO_MS}ms  temp={TEMPERATURE}  cfg={CFG_SCALE}\n")


# ── helpers ────────────────────────────────────────────────────────────────────

def _gcs_dir_exists(gcs_path: str) -> bool:
    return subprocess.run(["gsutil", "ls", gcs_path], capture_output=True).returncode == 0


def _gcs_cp(src: str, dst: str) -> None:
    subprocess.run(["gsutil", "-m", "cp", "-r", src, dst], check=True)


# ── 1. Download base model ─────────────────────────────────────────────────────

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


# ── 2. Load pipeline + adapter ─────────────────────────────────────────────────

def load_pipeline_with_adapter() -> HeartMuLaGenPipeline:
    print("Loading base HeartMuLa pipeline…")
    # Keep codec on CPU to save VRAM — it's only used for postprocessing (decode frames → waveform)
    pipe = HeartMuLaGenPipeline.from_pretrained(
        pretrained_path=str(CKPT_DIR),
        device={"mula": DEVICE, "codec": "cpu"},
        dtype={"mula": DTYPE, "codec": torch.float32},
        version=MODEL_SIZE,
    )

    print(f"Downloading LoRA adapter from {GCS_ADAPTER_PATH}…")
    # gsutil cp -r .../latest/adapter ./ckpt/ creates ./ckpt/adapter/
    _gcs_cp(GCS_ADAPTER_PATH, str(CKPT_DIR) + "/")
    local_adapter = CKPT_DIR / Path(GCS_ADAPTER_PATH).name   # e.g. "adapter"

    print("Applying LoRA adapter…")
    pipe.mula.backbone = PeftModel.from_pretrained(
        pipe.mula.backbone, str(local_adapter)
    )
    pipe.mula.backbone = pipe.mula.backbone.merge_and_unload()
    print("Adapter merged into backbone.")

    return pipe


# ── 3. Download input wav / tags if needed ────────────────────────────────────

def resolve_tags_file() -> Path | None:
    """Download GCS_TAGS_FILE to a local path and return it, or None if unset."""
    if not GCS_TAGS_FILE:
        return None
    local = Path("./data/paired/tags/input.txt")
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tags file from {GCS_TAGS_FILE}…")
    subprocess.run(["gsutil", "cp", GCS_TAGS_FILE, str(local)], check=True)
    print(f"  Tags : {local.read_text().strip()}")
    return local



def resolve_input_wav() -> str:
    """Return a local path to the input wav, downloading from GCS if needed."""
    if not INPUT_WAV:
        return ""
    if INPUT_WAV.startswith("gs://"):
        local = Path("./data/input.wav")
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading input wav from {INPUT_WAV}…")
        subprocess.run(["gsutil", "cp", INPUT_WAV, str(local)], check=True)
        return str(local)
    return INPUT_WAV


# ── 4. Generate ────────────────────────────────────────────────────────────────

def generate_clip(
    pipe: HeartMuLaGenPipeline,
    input_wav: str,
    idx: int,
    tags_path: Path | None = None,
    lyrics_path: Path | None = None,
) -> dict:
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


def postprocess_and_save(pipe: HeartMuLaGenPipeline, model_outputs: dict, idx: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"generated_{idx + 1:03d}.wav"

    # postprocess(outputs, save_path) saves the file and returns the path
    pipe.postprocess(model_outputs, str(out_path))

    print(f"  Saved → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    download_checkpoints()
    pipe = load_pipeline_with_adapter()
    pipe.mula.eval()

    input_wav = resolve_input_wav()
    tags_path = resolve_tags_file()

    if input_wav:
        print("\nAnnotating input wav…")
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
        out_path = postprocess_and_save(pipe, model_outputs, i)
        output_files.append(out_path)
        # Upload immediately so partial results are preserved on failure
        print(f"  Uploading {out_path.name} to GCS…")
        _gcs_cp(str(out_path), GCS_OUTPUT_PATH + "/")
        # Free VRAM between clips to avoid OOM on T4
        torch.cuda.empty_cache()

    print(f"\nAll {len(output_files)} clip(s) uploaded to {GCS_OUTPUT_PATH}.")
    print("Done.")


if __name__ == "__main__":
    main()
