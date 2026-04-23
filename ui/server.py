"""Bia Music Composer — UI server (runs locally or on Cloud Run).

Local dev (plain venv, no uv):
    pip install -r ui/requirements.txt
    python -m ui.server

Cloud Run:
    bash gcp/deploy_ui.sh
"""

from __future__ import annotations

import io
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import google.api_core.exceptions
from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from google.cloud import compute_v1, storage

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
UI_DIR = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Bia Music Composer")

# ── Single-job in-memory state (local single-user tool) ──────────────────────
_job: dict = {
    "status": "idle",  # idle | uploading | uploaded | launching | generating | complete | error
    "message": "Ready.",
    "input_gcs": None,
    "tags_gcs": None,
    "outputs": [],       # list of GCS blob names
    "started_at": None,  # ISO timestamp — used to filter outputs from this run
    "num_outputs": 1,    # how many outputs were requested
    "error": None,
}
_lock = threading.Lock()


def _cfg(key: str, default: Optional[str] = None) -> str:
    """Read from .env file if present (local dev), otherwise os.environ (Cloud Run)."""
    dot = dotenv_values(PROJECT_ROOT / ".env")
    val = dot.get(key) or os.getenv(key, default)
    if val is None:
        raise ValueError(f"Missing required config: {key}. Set it in .env or as an env var.")
    return val


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(UI_DIR / "index.html"))


@app.post("/api/upload")
async def upload(
    audio: UploadFile = File(...),
    tags: str = Form(...),
):
    """Upload audio + tags to GCS. Returns the two GCS paths."""
    with _lock:
        _job.update(
            status="uploading",
            message="Uploading audio and tags to GCS…",
            input_gcs=None, tags_gcs=None,
            outputs=[], started_at=None, error=None,
        )

    bucket_name = _cfg("GCS_BUCKET_NAME")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    gcs_client = storage.Client()
    bkt = gcs_client.bucket(bucket_name)

    # Upload audio
    audio_bytes = await audio.read()
    audio_gcs_key = f"data/ui/inputs/ui_{ts}.wav"
    bkt.blob(audio_gcs_key).upload_from_string(audio_bytes, content_type="audio/wav")
    input_gcs = f"gs://{bucket_name}/{audio_gcs_key}"

    # Upload tags
    tags_gcs_key = f"data/ui/tags/ui_{ts}.txt"
    bkt.blob(tags_gcs_key).upload_from_string(tags.strip(), content_type="text/plain")
    tags_gcs = f"gs://{bucket_name}/{tags_gcs_key}"

    with _lock:
        _job.update(
            status="uploaded",
            message="Uploaded. Ready to generate.",
            input_gcs=input_gcs,
            tags_gcs=tags_gcs,
        )

    return {"input_gcs": input_gcs, "tags_gcs": tags_gcs}


_ZONES = [
    "europe-west1-b", "europe-west1-c", "europe-west1-d",
    "europe-west3-b", "europe-west3-c",
    "europe-west4-a", "europe-west4-b", "europe-west4-c",
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "us-east1-b", "us-east1-c", "us-east4-a",
    "us-west1-a", "us-west1-b", "us-west4-a",
]


def _launch_vm(
    input_wav: str,
    tags_file: str,
    num_outputs: int,
    temperature: float,
    cfg_scale: float,
) -> None:
    """Background thread: create GCP SPOT VM via Compute Engine SDK (no gcloud CLI needed)."""
    try:
        project_id   = _cfg("PROJECT_ID")
        bucket_name  = _cfg("GCS_BUCKET_NAME")
        prefix       = _cfg("GCS_BUCKET_FOLDER_PREFIX", "heartmula")
        model_size   = _cfg("MODEL_SIZE", "3b").lower()
        run_mode     = _cfg("RUN_MODE", "test")
        docker_image = _cfg("DOCKER_IMAGE")
        wavlm_model  = _cfg("WAVLM_MODEL", "microsoft/wavlm-base")
        num_prefix   = _cfg("NUM_PREFIX_TOKENS", "32")
        max_audio_ms = _cfg("MAX_AUDIO_MS", "30000")
        vm_zone      = _cfg("VM_ZONE", "europe-west1-b")
        instance     = "heartmula-generate-a2a"

        # Read startup script (bundled in Docker image at /app/gcp/ or local project root)
        startup_script_path = PROJECT_ROOT / "gcp" / "generate_audio2audio_startup.sh"
        startup_script = startup_script_path.read_text()

        instances = compute_v1.InstancesClient()

        # Delete old instance (best-effort)
        try:
            instances.delete(project=project_id, zone=vm_zone, instance=instance).result(timeout=60)
        except (google.api_core.exceptions.NotFound, Exception):
            pass

        metadata_items = [
            compute_v1.Items(key="startup-script",            value=startup_script),
            compute_v1.Items(key="GCS_BUCKET_NAME",           value=bucket_name),
            compute_v1.Items(key="GCS_BUCKET_FOLDER_PREFIX",  value=prefix),
            compute_v1.Items(key="MODEL_SIZE",                 value=model_size),
            compute_v1.Items(key="RUN_MODE",                   value=run_mode),
            compute_v1.Items(key="INPUT_WAV",                  value=input_wav),
            compute_v1.Items(key="GCS_TAGS_FILE",              value=tags_file),
            compute_v1.Items(key="NUM_OUTPUTS",                value=str(num_outputs)),
            compute_v1.Items(key="MAX_AUDIO_MS",               value=max_audio_ms),
            compute_v1.Items(key="TEMPERATURE",                value=str(temperature)),
            compute_v1.Items(key="CFG_SCALE",                  value=str(cfg_scale)),
            compute_v1.Items(key="DOCKER_IMAGE",               value=docker_image),
            compute_v1.Items(key="WAVLM_MODEL",                value=wavlm_model),
            compute_v1.Items(key="NUM_PREFIX_TOKENS",          value=num_prefix),
        ]

        instance_resource = compute_v1.Instance(
            name=instance,
            machine_type=f"zones/ZONE/machineTypes/g2-standard-8",  # ZONE substituted per attempt
            disks=[compute_v1.AttachedDisk(
                boot=True,
                auto_delete=True,
                initialize_params=compute_v1.AttachedDiskInitializeParams(
                    source_image=(
                        "projects/deeplearning-platform-release/global/images/family/"
                        "pytorch-2-9-cu129-ubuntu-2404-nvidia-580"
                    ),
                    disk_size_gb=100,
                ),
            )],
            network_interfaces=[compute_v1.NetworkInterface(
                access_configs=[compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")],
            )],
            guest_accelerators=[compute_v1.AcceleratorConfig(
                accelerator_count=1,
                accelerator_type="zones/ZONE/acceleratorTypes/nvidia-l4",  # ZONE substituted
            )],
            scheduling=compute_v1.Scheduling(
                on_host_maintenance="TERMINATE",
                provisioning_model="STANDARD",
            ),
            metadata=compute_v1.Metadata(items=metadata_items),
            service_accounts=[compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )],
        )

        for zone in _ZONES:
            with _lock:
                _job["message"] = f"Trying zone {zone}…"

            # Substitute zone into machine_type and accelerator_type
            ir = compute_v1.Instance()
            ir._pb.CopyFrom(instance_resource._pb)
            ir.machine_type = f"zones/{zone}/machineTypes/g2-standard-8"
            ir.guest_accelerators[0].accelerator_type = f"zones/{zone}/acceleratorTypes/nvidia-l4"

            try:
                try:
                    instances.delete(project=project_id, zone=zone, instance=instance).result(timeout=60)
                except Exception:
                    pass
                op = instances.insert(project=project_id, zone=zone, instance_resource=ir)
                op.result(timeout=120)
                with _lock:
                    _job.update(
                        status="generating",
                        message=f"VM running in {zone}. Generation takes ~10–20 min.",
                    )
                return
            except Exception:
                continue  # Try next zone

        with _lock:
            _job.update(
                status="error",
                error="Could not create VM in any zone.",
                message="VM launch failed — no capacity found.",
            )

    except Exception as exc:
        with _lock:
            _job.update(status="error", error=str(exc), message=f"Error: {exc}")


@app.post("/api/generate")
async def generate(
    num_outputs: int = Form(3),
    temperature: float = Form(1.0),
    cfg_scale: float = Form(3.0),
):
    """Trigger GCP SPOT VM generation. /api/upload must be called first."""
    with _lock:
        input_gcs = _job.get("input_gcs")
        tags_gcs  = _job.get("tags_gcs") or ""

    if not input_gcs:
        raise HTTPException(400, "No audio uploaded. Call /api/upload first.")

    with _lock:
        _job.update(
            status="launching",
            message="Launching GCP SPOT VM…",
            started_at=datetime.now(timezone.utc).isoformat(),
            outputs=[],
            num_outputs=num_outputs,
        )

    threading.Thread(
        target=_launch_vm,
        args=(input_gcs, tags_gcs, num_outputs, temperature, cfg_scale),
        daemon=True,
    ).start()

    return {"status": "launching"}


@app.get("/api/status")
async def status():
    """Return current job state. Polls GCS for outputs in all non-error states."""
    with _lock:
        current = dict(_job)

    if current["status"] not in ("uploading", "error"):
        try:
            bucket_name = _cfg("GCS_BUCKET_NAME")
            prefix      = _cfg("GCS_BUCKET_FOLDER_PREFIX", "heartmula")
            model_size  = _cfg("MODEL_SIZE", "3b").lower()
            run_mode    = _cfg("RUN_MODE", "test")

            gcs_client = storage.Client()
            bkt = gcs_client.bucket(bucket_name)
            output_prefix = f"{prefix}-{model_size}/{run_mode}/latest/generated/"

            started_at = (
                datetime.fromisoformat(current["started_at"])
                if current.get("started_at") else None
            )

            wavs = [
                b for b in bkt.list_blobs(prefix=output_prefix)
                if b.name.endswith(".wav") and (
                    started_at is None
                    or b.updated.replace(tzinfo=timezone.utc) > started_at
                )
            ]

            if wavs:
                outputs = sorted(b.name for b in wavs)
                expected = current.get("num_outputs", 1)
                done = len(outputs) >= expected
                new_status = "complete" if done else current["status"]
                new_msg = (
                    f"Done — {len(outputs)} output(s) ready."
                    if done
                    else f"Generating… {len(outputs)}/{expected} output(s) ready."
                )
                with _lock:
                    _job.update(outputs=outputs, status=new_status, message=new_msg)
                current.update(outputs=outputs, status=new_status, message=new_msg)
        except Exception:
            pass  # Don't fail a status check

    return current


@app.get("/api/output/{blob_name:path}")
async def get_output(blob_name: str, dl: bool = False):
    """Serve a generated WAV from GCS. ?dl=1 forces a file download."""
    bucket_name = _cfg("GCS_BUCKET_NAME")
    gcs_client = storage.Client()
    blob = gcs_client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        raise HTTPException(404, "File not found in GCS.")
    data = blob.download_as_bytes()
    filename = blob_name.rsplit("/", 1)[-1]
    disposition = "attachment" if dl else "inline"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="audio/wav",
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


@app.post("/api/reset")
async def reset():
    """Clear job state so a new generation can be started."""
    with _lock:
        _job.update(
            status="idle", message="Ready.",
            input_gcs=None, tags_gcs=None,
            outputs=[], started_at=None, error=None,
        )
    return {"status": "idle"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ui.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(UI_DIR)],
    )
