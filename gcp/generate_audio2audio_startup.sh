#!/bin/bash
# Startup script for the audio2audio generate VM.
# Pulls the trainer Docker image and runs generate_audio2audio.py.

set -e
trap 'sudo shutdown -h now' EXIT

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
METADATA_HEADER="Metadata-Flavor: Google"

GCS_BUCKET_NAME=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/GCS_BUCKET_NAME")
GCS_BUCKET_FOLDER_PREFIX=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/GCS_BUCKET_FOLDER_PREFIX")
MODEL_SIZE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/MODEL_SIZE")
RUN_MODE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/RUN_MODE")
INPUT_WAV=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/INPUT_WAV")
NUM_OUTPUTS=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/NUM_OUTPUTS")
MAX_AUDIO_MS=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/MAX_AUDIO_MS")
TEMPERATURE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/TEMPERATURE")
CFG_SCALE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/CFG_SCALE")
DOCKER_IMAGE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/DOCKER_IMAGE")
GCS_TAGS_FILE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/GCS_TAGS_FILE" || echo "")
WAVLM_MODEL=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/WAVLM_MODEL" || echo "microsoft/wavlm-base")
NUM_PREFIX_TOKENS=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/NUM_PREFIX_TOKENS" || echo "32")

echo "--- Installing Docker ---"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
fi

echo "--- Configuring nvidia-container-toolkit for Docker ---"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
sleep 5

echo "--- Authenticating Docker with Artifact Registry ---"
gcloud auth configure-docker europe-west6-docker.pkg.dev --quiet

echo "--- Pulling Docker image: $DOCKER_IMAGE ---"
docker pull "$DOCKER_IMAGE"

echo "--- Running generate_audio2audio.py ---"
docker run --rm --gpus all --ipc=host \
    -e GCS_BUCKET_NAME="$GCS_BUCKET_NAME" \
    -e GCS_BUCKET_FOLDER_PREFIX="$GCS_BUCKET_FOLDER_PREFIX" \
    -e MODEL_SIZE="$MODEL_SIZE" \
    -e RUN_MODE="$RUN_MODE" \
    -e INPUT_WAV="$INPUT_WAV" \
    -e NUM_OUTPUTS="$NUM_OUTPUTS" \
    -e MAX_AUDIO_MS="$MAX_AUDIO_MS" \
    -e TEMPERATURE="$TEMPERATURE" \
    -e CFG_SCALE="$CFG_SCALE" \
    -e GCS_TAGS_FILE="$GCS_TAGS_FILE" \
    -e WAVLM_MODEL="$WAVLM_MODEL" \
    -e NUM_PREFIX_TOKENS="$NUM_PREFIX_TOKENS" \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$DOCKER_IMAGE" \
    uv run python scripts/generate_audio2audio.py

echo "--- Generation complete. Shutting down. ---"
sudo shutdown -h now || true
