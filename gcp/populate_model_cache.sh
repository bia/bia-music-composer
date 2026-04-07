#!/bin/bash
# Creates a CPU VM near the GCS bucket to download HuggingFace models
# and cache them at gs://{GCS_BUCKET_NAME}/model-cache/.
# Run this once before training. The VM shuts itself down when done.

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

INSTANCE_NAME="heartmula-model-cache"
# Use the same region as the GCS bucket for maximum upload speed
ZONE="${DOCKER_REGION}-a"

echo "--- Checking if model cache already exists ---"
if gsutil -q ls "gs://${GCS_BUCKET_NAME}/model-cache/HeartMuLa-oss-${MODEL_SIZE^^}/" 2>/dev/null; then
    echo "Cache already exists at gs://${GCS_BUCKET_NAME}/model-cache/"
    echo "Delete it first if you want to refresh: gsutil -m rm -r gs://${GCS_BUCKET_NAME}/model-cache/"
    exit 0
fi

echo "--- Deleting existing cache VM (if any) ---"
gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet 2>/dev/null || true

echo "--- Creating model cache VM in $ZONE ---"
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n2-standard-4 \
    --boot-disk-size=100GB \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --scopes=cloud-platform \
    --metadata=GCS_BUCKET_NAME=${GCS_BUCKET_NAME},MODEL_SIZE=${MODEL_SIZE} \
    --metadata-from-file=startup-script=gcp/cache_startup.sh

echo ""
echo "Cache VM created. It will shut down automatically when done (~20-30 min)."
echo "Monitor progress:"
echo "  gcloud logging tail 'resource.type=gce_instance' --project=$PROJECT_ID --format='value(jsonPayload.message)'"
