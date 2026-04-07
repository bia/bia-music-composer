#!/bin/bash
# Launches the training VM and watches for preemption.
# If preempted, automatically relaunches (up to MAX_RETRIES times).
# Training checkpoints in GCS mean each restart resumes where it left off.

set -e

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

MAX_RETRIES=${MAX_RETRIES:-10}
POLL_INTERVAL=30   # seconds between status checks
attempt=1

launch_vm() {
    echo ""
    echo "=== Attempt $attempt / $MAX_RETRIES ==="
    bash gcp/run_on_vm.sh
}

wait_for_completion() {
    local zone
    zone=$(grep '^VM_ZONE=' .env | cut -d= -f2)
    local instance="heartmula-${RUN_MODE}"

    echo "Watching $instance in $zone (polling every ${POLL_INTERVAL}s)…"
    while true; do
        sleep $POLL_INTERVAL

        status=$(gcloud compute instances describe "$instance" \
            --zone="$zone" --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

        if [ "$status" = "TERMINATED" ] || [ "$status" = "NOT_FOUND" ]; then
            # Check if it was preempted
            preempted=$(gcloud logging read \
                "protoPayload.methodName=compute.instances.preempted \
                 AND protoPayload.resourceName:$instance" \
                --freshness=5m --limit=1 --format="value(timestamp)" 2>/dev/null)

            if [ -n "$preempted" ]; then
                echo "  ↯ Preempted at $preempted — relaunching…"
                return 1   # signal: preempted, retry
            else
                echo "  ✓ VM terminated normally — training complete."
                return 0   # signal: done
            fi
        fi

        echo "  [$(date -u +%H:%M:%S)] status=$status"
    done
}

# ── Main retry loop ────────────────────────────────────────────────────────
while [ $attempt -le $MAX_RETRIES ]; do
    launch_vm

    if wait_for_completion; then
        echo ""
        echo "Training finished successfully after $attempt attempt(s)."
        exit 0
    fi

    attempt=$((attempt + 1))
    if [ $attempt -le $MAX_RETRIES ]; then
        echo "  Waiting 10s before retry…"
        sleep 10
    fi
done

echo "ERROR: Reached max retries ($MAX_RETRIES). Training did not complete."
exit 1
