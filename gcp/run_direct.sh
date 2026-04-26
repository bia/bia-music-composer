#!/bin/bash
# Launch a SPOT VM to run fast-shot direct audio2audio fine-tuning.
# Uses WavLM feature alignment loss on (input, output) pairs — no proxy generation.
#
# Options (all via env or .env):
#   NUM_STEPS             Gradient steps        (default: 100)
#   LEARNING_RATE         LR for projection     (default: 1e-4)
#   BACKBONE_ALIGN        Run phase-2 backbone  (default: false)
#   BACKBONE_ALIGN_STEPS  Steps for phase 2     (default: 20)
#
# Usage:
#   bash gcp/run_direct.sh
#   NUM_STEPS=200 BACKBONE_ALIGN=true bash gcp/run_direct.sh

set -e

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

export TRAINING_SCRIPT=direct
export NUM_STEPS="${NUM_STEPS:-100}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export BACKBONE_ALIGN="${BACKBONE_ALIGN:-false}"
export BACKBONE_ALIGN_STEPS="${BACKBONE_ALIGN_STEPS:-20}"

echo "=== Fast-shot direct audio2audio fine-tuning ==="
echo "    NUM_STEPS=${NUM_STEPS}  LR=${LEARNING_RATE}"
echo "    BACKBONE_ALIGN=${BACKBONE_ALIGN} (steps=${BACKBONE_ALIGN_STEPS})"
echo ""

# Pass extra vars via COMMON_METADATA extension — run_on_vm.sh adds TRAINING_SCRIPT already.
# The extra direct-training vars go in via the metadata override below.
# We patch run_on_vm.sh's COMMON_METADATA at call time via env injection.

INSTANCE_NAME="heartmula-audio2audio-direct"

L4_ZONES=(
    europe-west1-b europe-west1-c europe-west1-d
    europe-west3-b europe-west3-c
    europe-west4-a europe-west4-b europe-west4-c
    us-central1-a us-central1-b us-central1-c
    us-east1-b us-east1-c us-east4-a
    us-west1-b us-west4-a
    asia-east1-a asia-east1-b
    asia-northeast1-a asia-northeast1-b
)

METADATA="install-gpu-driver=True,MODEL_SIZE=${MODEL_SIZE},RUN_MODE=${RUN_MODE},PROJECT_ID=${PROJECT_ID},DOCKER_REGION=${DOCKER_REGION},GCS_BUCKET_FOLDER_PREFIX=${GCS_BUCKET_FOLDER_PREFIX},GCS_BUCKET_NAME=${GCS_BUCKET_NAME},GCS_PAIRS_PATH=${GCS_PAIRS_PATH},GCS_TAGS_PATH=${GCS_TAGS_PATH},NUM_PREFIX_TOKENS=${NUM_PREFIX_TOKENS:-32},AUDIO_ENCODER_MODEL=${AUDIO_ENCODER_MODEL:-microsoft/wavlm-base},TRAINING_SCRIPT=direct,NUM_STEPS=${NUM_STEPS},BACKBONE_ALIGN=${BACKBONE_ALIGN},BACKBONE_ALIGN_STEPS=${BACKBONE_ALIGN_STEPS}"

echo "--- Deleting existing VM instance (if any) ---"
for zone in "${L4_ZONES[@]}"; do
    gcloud compute instances delete $INSTANCE_NAME --zone=$zone --quiet 2>/dev/null && break
done

STYLE_TAGS_FILE=$(mktemp)
echo -n "${STYLE_TAGS:-piano,ambient,reflective}" > "$STYLE_TAGS_FILE"

_try_create() {
    local zone=$1 machine=$2 accel=$3
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$zone \
        --machine-type=$machine \
        --boot-disk-size=200GB \
        --accelerator=type=$accel,count=1 \
        --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT \
        --scopes=cloud-platform \
        --metadata="$METADATA" \
        --metadata-from-file=startup-script=gcp/startup.sh,STYLE_TAGS="$STYLE_TAGS_FILE" 2>&1
}

echo "--- Creating SPOT VM: $INSTANCE_NAME ---"
CREATED_ZONE=""
GPU_TYPE=""
for zone in "${L4_ZONES[@]}"; do
    echo "  L4 $zone…"
    result=$(_try_create $zone g2-standard-8 nvidia-l4) || true
    if echo "$result" | grep -q "RUNNING"; then
        CREATED_ZONE=$zone; GPU_TYPE="L4"; echo "$result"; break
    else
        echo "  └─ $(echo "$result" | grep 'message:' | head -1)"
    fi
done


rm -f "$STYLE_TAGS_FILE"

if [ -z "$CREATED_ZONE" ]; then
    echo "ERROR: Could not create VM in any zone."
    exit 1
fi

sed -i '' "s/^VM_ZONE=.*/VM_ZONE=${CREATED_ZONE}/" .env
echo ""
echo "VM created: $GPU_TYPE in $CREATED_ZONE"
echo "Updated adapter will upload to: gs://${GCS_BUCKET_NAME}/${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE}/${RUN_MODE}/latest/adapter"
echo ""
echo "Monitor logs:"
echo "  gcloud logging tail 'resource.type=gce_instance' --project=$PROJECT_ID --format='value(jsonPayload.message)'"
