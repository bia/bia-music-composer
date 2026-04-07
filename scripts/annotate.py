#!/usr/bin/env python3
"""
Auto-annotate a paired input audio file with tags and lyrics.

Tags  : zero-shot audio classification via LAION CLAP
Lyrics: speech/singing transcription via OpenAI Whisper
         Language is inferred from an existing tags file when available.

Usage (standalone)
------------------
    python scripts/annotate.py data/paired/inputs/my_input.wav

Writes to:
    data/paired/tags/<stem>.txt
    data/paired/lyrics/<stem>.txt

The paired data root is inferred as the grandparent of the audio file
(inputs/ → paired/), or overridden with --data-dir.

Importable API
--------------
    from scripts.annotate import annotate_file
    tags_path, lyrics_path = annotate_file(audio_path)
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, ClapModel, WhisperForConditionalGeneration, WhisperProcessor, pipeline

# ---------------------------------------------------------------------------
# Tag vocabulary  (genre · instrument · mood · tempo · language)
# ---------------------------------------------------------------------------
TAG_CANDIDATES = [
    # genres
    "pop", "rock", "alternative", "indie", "jazz", "blues", "soul", "r&b",
    "neo-soul", "hip-hop", "rap", "electronic", "dance", "house", "techno",
    "ambient", "classical", "orchestral", "folk", "country", "bossa nova",
    "samba", "latin", "reggae", "afrobeats", "funk", "gospel",
    # instruments
    "piano", "guitar", "acoustic guitar", "electric guitar", "bass", "drums",
    "cajon", "strings", "violin", "cello", "brass", "trumpet", "saxophone",
    "flute", "synthesizer", "vocals", "choir", "ukulele",
    # mood / feel
    "ballad", "romantic", "melancholic", "uplifting", "energetic", "calm",
    "dreamy", "intense", "nostalgic", "dark", "playful", "emotional",
    # tempo
    "slow", "midtempo", "uptempo", "fast",
    # production
    "acoustic", "lo-fi", "produced", "live", "sparse", "lush",
    # language / cultural
    "english", "french", "portuguese", "spanish", "multilingual",
    "instrumental",
]

TOP_K = 8

# Map tag keywords → Whisper language codes
_LANGUAGE_TAG_MAP = {
    "portuguese": "pt",
    "french": "fr",
    "spanish": "es",
    "english": "en",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "mandarin": "zh",
    "arabic": "ar",
}


def _detect_language_from_tags(tags_file: Path) -> str | None:
    """Return a Whisper language code inferred from an existing tags file, or None."""
    if not tags_file.exists():
        return None
    tags_text = tags_file.read_text().lower()
    for keyword, code in _LANGUAGE_TAG_MAP.items():
        if keyword in tags_text:
            return code
    return None


def _load_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)  # (T,)


def _load_audio_mono_16k(path: Path) -> torch.Tensor:
    return _load_audio_mono(path, 16000)


def _generate_tags(audio_path: Path, device: str) -> str:
    print("  Loading CLAP model …")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    audio_np = _load_audio_mono(audio_path, 48000).numpy()

    inputs = processor(
        text=TAG_CANDIDATES,
        audios=[audio_np],
        return_tensors="pt",
        padding=True,
        sampling_rate=48000,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_audio[0]  # (num_tags,)
    probs = logits.softmax(dim=-1).cpu().tolist()

    ranked = sorted(zip(TAG_CANDIDATES, probs), key=lambda x: -x[1])
    top_tags = [tag for tag, _ in ranked[:TOP_K]]
    return ",".join(top_tags)


def _generate_lyrics(audio_path: Path, device: str, language: str | None) -> str:
    print("  Loading Whisper model …")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)

    if language:
        print(f"  Whisper language hint (from tags): {language}")
    else:
        # Detect language from first 30 s of audio
        waveform = _load_audio_mono_16k(audio_path)
        chunk = waveform[: 16000 * 30].numpy()
        features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            lang_ids = model.detect_language(features)
        language = processor.batch_decode(lang_ids)[0].strip("<|>")
        print(f"  Whisper detected language: {language}")

    # Reuse the already-loaded model inside the pipeline (no second load)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )

    result = asr(str(audio_path), generate_kwargs={"task": "transcribe", "language": language})
    text: str = result["text"].strip()

    if not text:
        return ""

    # Wrap in a [Verse] header to match the existing lyrics format
    return f"[Verse]\n{text}\n"


def annotate_file(
    audio_path: Path,
    data_dir: Path | None = None,
    overwrite_tags: bool = False,
    overwrite_lyrics: bool = False,
) -> tuple[Path, Path]:
    """
    Generate tags and lyrics for *audio_path*, writing results to the
    paired data directory.

    Returns (tags_path, lyrics_path).  Skips generation for files that
    already exist unless overwrite_* is True.
    """
    audio_path = Path(audio_path).resolve()
    stem = audio_path.stem

    if data_dir is None:
        data_dir = audio_path.parent.parent  # inputs/ → paired/

    tags_dir = Path(data_dir) / "tags"
    lyrics_dir = Path(data_dir) / "lyrics"
    tags_dir.mkdir(parents=True, exist_ok=True)
    lyrics_dir.mkdir(parents=True, exist_ok=True)

    tag_file = tags_dir / f"{stem}.txt"
    lyrics_file = lyrics_dir / f"{stem}.txt"

    need_tags = overwrite_tags or not tag_file.exists() or tag_file.read_text().strip() == ""
    need_lyrics = overwrite_lyrics or not lyrics_file.exists() or lyrics_file.read_text().strip() == ""

    if not need_tags and not need_lyrics:
        print(f"  Annotations already exist for {stem}, skipping.")
        return tag_file, lyrics_file

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if need_tags:
        print(f"\n=== Generating tags for {stem} ===")
        tags = _generate_tags(audio_path, device)
        tag_file.write_text(tags + "\n")
        print(f"  Tags : {tags}")
        print(f"  Wrote: {tag_file}")

    if need_lyrics:
        # Detect language from tags (freshly written or pre-existing)
        language = _detect_language_from_tags(tag_file)
        print(f"\n=== Generating lyrics for {stem} ===")
        lyrics = _generate_lyrics(audio_path, device, language)
        lyrics_file.write_text(lyrics)
        print(f"  Lyrics preview:\n{lyrics[:300]}")
        print(f"  Wrote: {lyrics_file}")

    return tag_file, lyrics_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-annotate paired audio file")
    parser.add_argument("audio", type=Path, help="Path to input wav file")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--overwrite-tags", action="store_true")
    parser.add_argument("--overwrite-lyrics", action="store_true")
    args = parser.parse_args()

    if not args.audio.exists():
        sys.exit(f"File not found: {args.audio}")

    annotate_file(
        args.audio,
        data_dir=args.data_dir,
        overwrite_tags=args.overwrite_tags,
        overwrite_lyrics=args.overwrite_lyrics,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
