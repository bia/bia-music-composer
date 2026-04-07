#!/bin/bash
# Startup script for the audio2audio finetune VM.
# Pulls the trainer Docker image and runs finetune_audio2audio.py.

set -e
trap 'sudo shutdown -h now' EXIT

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
METADATA_HEADER="Metadata-Flavor: Google"

GCS_BUCKET_NAME=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/GCS_BUCKET_NAME")
GCS_BUCKET_FOLDER_PREFIX=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/GCS_BUCKET_FOLDER_PREFIX")
MODEL_SIZE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/MODEL_SIZE")
RUN_MODE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/RUN_MODE")
DOCKER_IMAGE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/DOCKER_IMAGE")
NUM_EPOCHS=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/NUM_EPOCHS" || echo "3")
LORA_RANK=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/LORA_RANK" || echo "16")
LEARNING_RATE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/LEARNING_RATE" || echo "5e-5")
MAX_AUDIO_MS=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/MAX_AUDIO_MS" || echo "30000")
WAVLM_MODEL=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/WAVLM_MODEL" || echo "microsoft/wavlm-base")
NUM_PREFIX_TOKENS=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/NUM_PREFIX_TOKENS" || echo "32")

echo "--- Ensuring Docker is available ---"
if ! command -v docker &> /dev/null; then
    # Retry Docker install with backoff in case of transient network issues
    for attempt in 1 2 3; do
        echo "  Docker install attempt $attempt..."
        curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh && break
        sleep 15
    done
fi

echo "--- Configuring nvidia-container-toolkit for Docker ---"
nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
systemctl restart docker
sleep 5

echo "--- Authenticating Docker with Artifact Registry ---"
gcloud auth configure-docker europe-west6-docker.pkg.dev --quiet

echo "--- Pulling Docker image: $DOCKER_IMAGE ---"
docker pull "$DOCKER_IMAGE"

echo "--- Running finetune_audio2audio.py ---"
docker run --rm --gpus all --ipc=host \
    -e GCS_BUCKET_NAME="$GCS_BUCKET_NAME" \
    -e GCS_BUCKET_FOLDER_PREFIX="$GCS_BUCKET_FOLDER_PREFIX" \
    -e MODEL_SIZE="$MODEL_SIZE" \
    -e RUN_MODE="$RUN_MODE" \
    -e NUM_EPOCHS="$NUM_EPOCHS" \
    -e LORA_RANK="$LORA_RANK" \
    -e LEARNING_RATE="$LEARNING_RATE" \
    -e MAX_AUDIO_MS="$MAX_AUDIO_MS" \
    -e WAVLM_MODEL="$WAVLM_MODEL" \
    -e NUM_PREFIX_TOKENS="$NUM_PREFIX_TOKENS" \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$DOCKER_IMAGE" \
    bash -c "
        set -e
        export PYTHONPATH=.
        echo '--- Downloading paired audio from GCS ---'
        mkdir -p data/paired
        gsutil -m cp -r gs://${GCS_BUCKET_NAME}/data/paired/* data/paired/
        echo '--- Starting finetune_audio2audio.py ---'
        uv run python scripts/finetune_audio2audio.py
    "

echo "--- Fine-tuning complete. Shutting down. ---"
sudo shutdown -h now || true
