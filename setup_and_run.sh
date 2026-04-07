#!/bin/bash
set -e
# This script is the main entrypoint inside the Docker container for cloud training.

# Set environment variable to prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Add the project root to the PYTHONPATH to allow for absolute imports
export PYTHONPATH=.

# --- GCS Bucket Configuration ---
# These variables are passed from the `docker run` command.
MODEL_SIZE=${MODEL_SIZE,,}   # normalise to lowercase for GCS paths
GCS_BUCKET_NAME=${GCS_BUCKET_NAME}
GCS_BUCKET_FOLDER_PREFIX=${GCS_BUCKET_FOLDER_PREFIX}
GCS_DESTINATION="gs://${GCS_BUCKET_NAME}/${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE}/${RUN_MODE}"

# --- Test GCS Upload Permission ---
echo "--- Testing GCS upload permission ---"
DUMMY_FILE="gcs_permission_test.txt"
echo "This is a test file to check GCS write permissions." > $DUMMY_FILE

# Try to upload the dummy file. The `|| exit 1` will cause the script to exit if gsutil fails.
echo "Uploading test file to $GCS_DESTINATION..."
gsutil cp $DUMMY_FILE "${GCS_DESTINATION}/${DUMMY_FILE}" || exit 1

# If upload was successful, remove the dummy file from the bucket and the local filesystem.
echo "GCS permission test successful. Cleaning up test file..."
gsutil rm "${GCS_DESTINATION}/${DUMMY_FILE}"
rm $DUMMY_FILE

echo "--- GCS permission test passed. Proceeding with main script. ---"

KNOWLEDGE_DEST="${GCS_DESTINATION}/knowledge"

# --- Download paired audio for few-shot training (if present in GCS) ---
GCS_PAIRED_DATA="gs://${GCS_BUCKET_NAME}/data/paired"
if gsutil ls "${GCS_PAIRED_DATA}" &>/dev/null; then
    echo "--- Downloading paired audio from ${GCS_PAIRED_DATA} ---"
    mkdir -p data/paired
    gsutil -m cp -r "${GCS_PAIRED_DATA}/*" data/paired/
    echo "Paired audio ready at data/paired/"
fi

echo "--- Starting training ---"
uv run python scripts/finetune.py

echo "--- All training stages complete! ---"

# --- Upload results to GCS ---
echo "Uploading adapter models to GCS..."

# Define a destination for the new timestamped results
TIMESTAMP=$(date +%y-%m-%d-%H-%M-%S)
TIMESTAMP_DEST="${GCS_DESTINATION}/${TIMESTAMP}"

# 1. Upload the results to a unique, timestamped directory
echo "Uploading results to timestamped directory: ${TIMESTAMP_DEST}"
gsutil -m cp -r "out/adapters/multimodal" "${TIMESTAMP_DEST}/adapter"
gsutil -m cp -r "out/models/multimodal" "${TIMESTAMP_DEST}/model"

# --- Update the 'latest' pointer ---
LATEST_DEST="${GCS_DESTINATION}/latest"

# 2. Remove the old 'latest' directory. The `|| true` part
#    prevents the script from exiting if the directory doesn't exist.
echo "Removing old 'latest' directory..."
(gsutil -m rm -r "${LATEST_DEST}" || true)

# 3. Perform a fast, server-side copy from the new timestamped directory
#    to create the new 'latest' directory.
echo "Updating 'latest' to point to new results by copying from ${TIMESTAMP_DEST}"
gsutil -m cp -r "${TIMESTAMP_DEST}" "${LATEST_DEST}"
