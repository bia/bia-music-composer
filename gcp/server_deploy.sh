#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Load Configuration from .env file ---
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# --- Configuration ---
# Define names for the repository and image
# We use a distinct repository/image name for the inference server to keep it separate from training artifacts
REPO_NAME="edugraph-predict"
IMAGE_NAME="edugraph-predict"
# The full image tag
IMAGE_TAG="$DOCKER_REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

# The Cloud Run service name
SERVICE_NAME="edugraph-predict"

# 1. Create Artifact Registry repository (if it doesn't exist)
echo "--- Checking for Artifact Registry repository: $REPO_NAME ---"
if ! gcloud artifacts repositories describe $REPO_NAME --location=$DOCKER_REGION --project=$PROJECT_ID &> /dev/null; then
    echo "Repository not found. Creating..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$DOCKER_REGION \
        --description="Docker repository for EduGraph Server"
else
    echo "Repository already exists."
fi

# 2. Configure Docker authentication
echo "--- Configuring Docker to authenticate with GCP... ---"
gcloud auth configure-docker $DOCKER_REGION-docker.pkg.dev

# 3. Build and Push the Docker image
echo "--- Building the Docker image: $IMAGE_TAG ---"
# Build from the server directory (assuming script is run from project root)
docker build -t $IMAGE_TAG ./server

echo "--- Pushing the Docker image to Artifact Registry... ---"
docker push $IMAGE_TAG

# 4. Deploy/Update Cloud Run Service
echo "--- Updating Cloud Run service: $SERVICE_NAME ---"
# Explicitly setting resources to match the existing high-performance/GPU config
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_TAG \
    --region $DOCKER_REGION \
    --project $PROJECT_ID \
    --cpu 4 \
    --memory 16Gi \
    --no-cpu-throttling \
    --cpu-boost \
    --gpu-type nvidia-l4 \
    --no-gpu-zonal-redundancy \
    --min-instances 0 \
    --max-instances 1 \
    --port 8080 \
    --concurrency 4 \
    --timeout 300 \
    --service-account $SERVICE_ACCOUNT

echo "--- Deployment command finished. ---"
