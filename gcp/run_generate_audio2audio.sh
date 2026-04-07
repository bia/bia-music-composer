#!/bin/bash
# Launch a SPOT VM to run generate_audio2audio.py (WavLM-conditioned generation).
# Usage: bash gcp/run_generate_audio2audio.sh

set -e

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

INSTANCE_NAME="heartmula-generate-a2a"
MODEL_SIZE_LOWER=$(echo "${MODEL_SIZE}" | tr '[:upper:]' '[:lower:]')
GCS_OUTPUT="gs://${GCS_BUCKET_NAME}/${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE_LOWER}/${RUN_MODE}/latest/generated"

ZONES=(
    europe-west1-b europe-west1-c europe-west1-d
    europe-west3-b europe-west3-c
    europe-west4-a europe-west4-b europe-west4-c
    us-central1-a us-central1-b us-central1-c us-central1-f
    us-east1-b us-east1-c us-east4-a
    us-west1-a us-west1-b us-west4-a
)

echo "--- Deleting existing VM instance (if any) ---"
gcloud compute instances delete $INSTANCE_NAME \
    --zone="${VM_ZONE:-europe-west1-b}" --quiet 2>/dev/null || true

echo "--- Creating SPOT VM: $INSTANCE_NAME (audio2audio) ---"
CREATED_ZONE=""
for zone in "${ZONES[@]}"; do
    echo "  Trying $zone..."
    result=$(gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$zone \
        --machine-type=g2-standard-8 \
        --accelerator=type=nvidia-l4,count=1 \
        --maintenance-policy=TERMINATE \
        --provisioning-model=STANDARD \
        --boot-disk-size=100GB \
        --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
        --image-project=deeplearning-platform-release \
        --scopes=cloud-platform \
        --metadata=GCS_BUCKET_NAME=${GCS_BUCKET_NAME},GCS_BUCKET_FOLDER_PREFIX=${GCS_BUCKET_FOLDER_PREFIX},MODEL_SIZE=${MODEL_SIZE_LOWER},RUN_MODE=${RUN_MODE},NUM_OUTPUTS=${NUM_OUTPUTS:-3},MAX_AUDIO_MS=${MAX_AUDIO_MS:-30000},TEMPERATURE=${TEMPERATURE:-1.0},CFG_SCALE=${CFG_SCALE:-3.0},DOCKER_IMAGE=${DOCKER_IMAGE},INPUT_WAV=${INPUT_WAV},GCS_TAGS_FILE=${GCS_TAGS_FILE:-},WAVLM_MODEL=${WAVLM_MODEL:-microsoft/wavlm-base},NUM_PREFIX_TOKENS=${NUM_PREFIX_TOKENS:-32} \
        --metadata-from-file=startup-script=gcp/generate_audio2audio_startup.sh \
        2>&1) || true

    if echo "$result" | grep -q "RUNNING"; then
        CREATED_ZONE=$zone
        echo "$result"
        break
    fi
    echo "  Failed in $zone, trying next…"
done

if [ -z "$CREATED_ZONE" ]; then
    echo "ERROR: Could not create VM in any zone."
    exit 1
fi

echo ""
echo "VM created in zone: $CREATED_ZONE"
echo "Output will appear at: $GCS_OUTPUT"
echo ""
echo "Monitor logs:"
echo "  gcloud logging tail 'resource.type=gce_instance' --project=$PROJECT_ID --format='value(jsonPayload.message)'"
