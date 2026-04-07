#!/bin/bash
# Startup script for the model-cache VM.
# Downloads HeartMuLa models from HuggingFace and uploads them to GCS.
# Runs once, then the VM shuts itself down.

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
METADATA_HEADER="Metadata-Flavor: Google"
GCS_BUCKET_NAME=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/GCS_BUCKET_NAME")
MODEL_SIZE=$(curl -s -H "$METADATA_HEADER" "$METADATA_URL/MODEL_SIZE")
MODEL_SIZE_UPPER=$(echo "$MODEL_SIZE" | tr '[:lower:]' '[:upper:]')

GCS_CACHE="gs://${GCS_BUCKET_NAME}/model-cache"

echo "--- Installing pip and huggingface_hub ---"
apt-get update -qq && apt-get install -y python3-pip -qq
pip3 install -q huggingface_hub

mkdir -p /tmp/models

# ── HeartMuLaGen shared assets ─────────────────────────────────────────────
echo "--- Downloading HeartMuLaGen from HuggingFace ---"
huggingface-cli download HeartMuLa/HeartMuLaGen --local-dir /tmp/models/HeartMuLaGen
echo "--- Uploading HeartMuLaGen to GCS ---"
gsutil -m rsync -r -x '\.cache' /tmp/models/HeartMuLaGen "${GCS_CACHE}/HeartMuLaGen"
rm -rf /tmp/models/HeartMuLaGen

# ── HeartMuLa main model ───────────────────────────────────────────────────
echo "--- Downloading HeartMuLa-oss-${MODEL_SIZE_UPPER} from HuggingFace ---"
huggingface-cli download HeartMuLa/HeartMuLa-oss-3B-happy-new-year \
    --local-dir "/tmp/models/HeartMuLa-oss-${MODEL_SIZE_UPPER}"
echo "--- Uploading HeartMuLa-oss-${MODEL_SIZE_UPPER} to GCS ---"
gsutil -m rsync -r -x '\.cache' \
    "/tmp/models/HeartMuLa-oss-${MODEL_SIZE_UPPER}" \
    "${GCS_CACHE}/HeartMuLa-oss-${MODEL_SIZE_UPPER}"
rm -rf "/tmp/models/HeartMuLa-oss-${MODEL_SIZE_UPPER}"

# ── HeartCodec ─────────────────────────────────────────────────────────────
echo "--- Downloading HeartCodec-oss from HuggingFace ---"
huggingface-cli download HeartMuLa/HeartCodec-oss-20260123 \
    --local-dir /tmp/models/HeartCodec-oss
echo "--- Uploading HeartCodec-oss to GCS ---"
gsutil -m rsync -r -x '\.cache' /tmp/models/HeartCodec-oss "${GCS_CACHE}/HeartCodec-oss"
rm -rf /tmp/models/HeartCodec-oss

echo "--- Model cache complete. Shutting down. ---"
sudo shutdown -h now
