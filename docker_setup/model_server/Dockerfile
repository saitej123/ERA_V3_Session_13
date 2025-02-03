# Use NVIDIA CUDA development image instead of runtime
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
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install flash-attention
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation

# Copy model files and server code
COPY server.py .
COPY model/ /app/model/

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"] 