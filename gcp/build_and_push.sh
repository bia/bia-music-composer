#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Load Configuration from .env file ---
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# --- Configuration from environment ---
# These variables are now loaded from the .env file.

# Define names for the repository and image
REPO_NAME="${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE}"
IMAGE_NAME="heartmula-trainer"
IMAGE_TAG="$DOCKER_REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

# 1. Create Artifact Registry repository (if it doesn't exist)
echo "--- Checking for Artifact Registry repository: $REPO_NAME ---"
if ! gcloud artifacts repositories describe $REPO_NAME --location=$DOCKER_REGION --project=$PROJECT_ID &> /dev/null; then
    echo "Repository not found. Creating..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$DOCKER_REGION \
        --description="Docker repository for Qwen training"
else
    echo "Repository already exists."
fi

# 2. Configure Docker authentication
echo "--- Configuring Docker to authenticate with GCP... ---"
gcloud auth configure-docker $DOCKER_REGION-docker.pkg.dev

# 3. Build and Push the Docker image
echo "--- Building the Docker image: $IMAGE_TAG ---"
# Pass the MODEL_SIZE as a build-arg to the Dockerfile
docker build --platform linux/amd64 --build-arg MODEL_SIZE=$MODEL_SIZE -t $IMAGE_TAG .

echo "--- Pushing the Docker image to Artifact Registry... "
docker push $IMAGE_TAG

