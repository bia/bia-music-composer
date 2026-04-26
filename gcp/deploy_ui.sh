#!/bin/bash
# Build and deploy the Bia Music Composer UI to Cloud Run.
# Usage: bash gcp/deploy_ui.sh
#
# Requires: gcloud CLI authenticated, .env file present.
# The service runs as the Compute Engine default service account, which needs:
#   - roles/storage.objectAdmin  (GCS uploads + output reads)
#   - roles/compute.instanceAdmin.v1  (launch SPOT VMs)
#   - roles/iam.serviceAccountUser   (attach service account to VM)

set -e

if [ ! -f .env ]; then
    echo "Error: .env file not found. Run from the project root."
    exit 1
fi
export $(grep -v '^#' .env | xargs)

SERVICE_NAME="bia-music-composer-ui"
REGION="${DOCKER_REGION:-europe-west6}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/heartmula-3b/${SERVICE_NAME}:latest"

echo "--- Building UI image ---"
docker build -f ui/Dockerfile -t "$IMAGE" .

echo "--- Pushing to Artifact Registry ---"
docker push "$IMAGE"

echo "--- Deploying to Cloud Run (region: $REGION) ---"
gcloud run deploy "$SERVICE_NAME" \
    --image="$IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --no-allow-unauthenticated \
    --memory=512Mi \
    --cpu=1 \
    --timeout=60 \
    --set-env-vars="\
PROJECT_ID=${PROJECT_ID},\
GCS_BUCKET_NAME=${GCS_BUCKET_NAME},\
GCS_BUCKET_FOLDER_PREFIX=${GCS_BUCKET_FOLDER_PREFIX},\
MODEL_SIZE=${MODEL_SIZE:-3b},\
RUN_MODE=${RUN_MODE:-test},\
DOCKER_IMAGE=${DOCKER_IMAGE},\
VM_ZONE=${VM_ZONE:-europe-west4-a},\
AUDIO_ENCODER_MODEL=${AUDIO_ENCODER_MODEL:-microsoft/wavlm-base},\
NUM_PREFIX_TOKENS=${NUM_PREFIX_TOKENS:-32},\
MAX_AUDIO_MS=${MAX_AUDIO_MS:-30000},\
TEMPERATURE=${TEMPERATURE:-1.0},\
CFG_SCALE=${CFG_SCALE:-3.0}"

echo ""
echo "--- Deployed. Open in browser: ---"
gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --format="value(status.url)"
echo ""
echo "To open with your Google account:"
echo "  gcloud run services proxy $SERVICE_NAME --region=$REGION --port=8080"
echo "  Then visit http://localhost:8080"
