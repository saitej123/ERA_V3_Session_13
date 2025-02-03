# Use NVIDIA CUDA development image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention dependencies
RUN pip3 install --no-cache-dir packaging ninja

# Install Flash Attention directly from the repository to avoid pip issues
RUN git clone --recursive https://github.com/Dao-AILab/flash-attention.git /app/flash-attention && \
    cd /app/flash-attention && \
    pip3 install .

# Copy model files and server code
COPY server.py .
COPY model/ /app/model/

# Expose port for FastAPI server
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
