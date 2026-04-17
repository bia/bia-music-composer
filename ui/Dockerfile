FROM python:3.12-slim

WORKDIR /app

# Install UI dependencies only (no ML stack)
COPY ui/requirements.txt /app/ui/requirements.txt
RUN pip install --no-cache-dir -r /app/ui/requirements.txt

# Copy UI source and the startup script the server reads at runtime
COPY ui/ /app/ui/
COPY gcp/generate_audio2audio_startup.sh /app/gcp/generate_audio2audio_startup.sh

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "ui.server:app", "--host", "0.0.0.0", "--port", "8080"]
