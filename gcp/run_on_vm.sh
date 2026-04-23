#!/bin/bash

# This script creates the VM directly using gcloud compute.
# It cycles through zones automatically until a SPOT instance is created.

# --- Load Configuration from .env file ---
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID not found in .env file"
    exit 1
fi

INSTANCE_NAME="heartmula-${RUN_MODE}"

ZONES=(
    europe-west1-b
    europe-west1-c
    europe-west1-d
    europe-west3-b
    europe-west3-c
    europe-west4-a
    europe-west4-b
    europe-west4-c
    us-central1-a
    us-central1-b
    us-central1-c
    us-east1-b
    us-east1-c
    us-west1-b
    us-west4-a
    us-east4-a
)

# --- Delete existing instance from whichever zone it's in ---
echo "--- Deleting existing VM instance (if any) ---"
for zone in "${ZONES[@]}"; do
    gcloud compute instances delete $INSTANCE_NAME --zone=$zone --quiet 2>/dev/null && break
done

# --- Try each zone until one succeeds ---
echo "--- Creating SPOT VM: $INSTANCE_NAME (cycling zones) ---"
CREATED_ZONE=""
for zone in "${ZONES[@]}"; do
    echo "  Trying $zone..."
    result=$(gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$zone \
        --machine-type=g2-standard-8 \
        --boot-disk-size=200GB \
        --accelerator=type=nvidia-l4,count=1 \
        --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --provisioning-model=STANDARD \
        --scopes=cloud-platform \
        --metadata=install-gpu-driver=True,google-monitoring-enabled=true,MODEL_SIZE=${MODEL_SIZE},RUN_MODE=${RUN_MODE},PROJECT_ID=${PROJECT_ID},DOCKER_REGION=${DOCKER_REGION},GCS_BUCKET_FOLDER_PREFIX=${GCS_BUCKET_FOLDER_PREFIX},GCS_BUCKET_NAME=${GCS_BUCKET_NAME},NUM_EPOCHS=${NUM_EPOCHS:-3},LORA_RANK=${LORA_RANK:-16} \
        --metadata-from-file=startup-script=gcp/startup.sh 2>&1)
    if echo "$result" | grep -q "RUNNING"; then
        CREATED_ZONE=$zone
        echo "$result"
        break
    else
        echo "  Failed: $(echo "$result" | grep 'message:' | head -1)"
    fi
done

if [ -z "$CREATED_ZONE" ]; then
    echo "ERROR: Could not create VM in any zone. All zones exhausted."
    exit 1
fi

echo ""
echo "VM created in zone: $CREATED_ZONE"

# Update VM_ZONE in .env so other scripts (e.g. download_results.sh) know where to look
sed -i '' "s/^VM_ZONE=.*/VM_ZONE=${CREATED_ZONE}/" .env
echo "Updated VM_ZONE=${CREATED_ZONE} in .env"
