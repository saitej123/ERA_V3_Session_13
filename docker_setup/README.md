# Docker Setup for SmolLM2 Model

This setup consists of two Docker containers:
1. A model server that serves the SmolLM2 model via FastAPI
2. A client with a Gradio interface that communicates with the model server

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime installed
- NVIDIA GPU with CUDA support
- The trained model checkpoint in the `model/checkpoints` directory
- The model configuration file (`config.yaml`)

## Installation

1. Install Docker:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

2. Install NVIDIA Docker runtime:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Directory Structure

```
docker_setup/
├── model_server/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── server.py
├── client/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── client.py
├── docker-compose.yml
└── README.md
```

## Usage

1. Make sure your trained model checkpoint is in the correct location:
   - The checkpoint should be at `../model/checkpoints/step_5051.pt`
   - The config file should be at `../config.yaml`

2. Start the containers:
```bash
docker-compose up --build
```

3. Access the interfaces:
   - Gradio client interface: http://localhost:7860
   - FastAPI documentation: http://localhost:8000/docs

4. To stop the containers:
```bash
docker-compose down
```

## Troubleshooting

1. If the model server fails to start:
   - Check if NVIDIA runtime is properly installed
   - Verify GPU is available with `nvidia-smi`
   - Check if model checkpoint and config files are in the correct location

2. If the client can't connect to the server:
   - Wait for the model server to fully initialize
   - Check if both containers are running with `docker-compose ps`
   - Check the logs with `docker-compose logs`

## Recording Instructions

To create a video demonstration:

1. Start screen recording
2. Show the docker containers starting up:
   ```bash
   docker-compose up --build
   ```
3. In another terminal, show the running containers:
   ```bash
   docker-compose ps
   ```
4. Open the Gradio interface in your browser (http://localhost:7860)
5. Enter a prompt and show the model generating text
6. Show the FastAPI documentation (http://localhost:8000/docs)
7. Stop the containers:
   ```bash
   docker-compose down
   ```

Upload the video to YouTube (can be unlisted) and share the link. 