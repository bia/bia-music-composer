# Use an official NVIDIA CUDA base image - this version is compatible with our torch build
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

# Install all system dependencies in one layer for better caching
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- Dependency Layer ---
# Install Google Cloud SDK for gsutil.
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/usr/local
ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# Copy the project definition file.
COPY pyproject.toml uv.lock .

RUN uv sync

# Finally, copy the actual application code.
COPY . .

# Define the default command that will be run when the container starts
CMD ["bash", "setup_and_run.sh"]
