# Triton-Jina-v3-TRT Setup Guide

This guide provides detailed instructions for setting up the Triton Inference Server with Jina Embeddings v3 using NVIDIA's TensorRT (TRT) on a Linux system with Docker and NVIDIA GPUs. Follow these steps to install prerequisites, authenticate with NVIDIA NGC, download model weights, configure the Triton model repository, and manage the service using `systemd`.

- [**Installation**](#installation)
    - [Step 1: System Prerequisites](#step-1-system-prerequisites)
        - [1.1 Install Docker](#11-install-docker)
        - [1.2 Install NVIDIA Container Toolkit](#12-install-nvidia-container-toolkit)
    - [Step 2: NVIDIA NGC Authentication](#step-2-nvidia-ngc-authentication)
        - [2.1 Get Your NGC API Key](#21-get-your-ngc-api-key)
        - [2.2 Log in via Docker](#22-log-in-via-docker)
    - [Step 3: Download Jina Embeddings v3 Model](#step-3-download-jina-embeddings-v3-model)
    - [Step 4: Pull Triton Docker Image](#step-4-pull-triton-docker-image)
    - [Step 5: Prepare Triton Model Repository](#step-5-prepare-triton-model-repository)
        - [5.1 Move the ONNX Model](#51-move-the-onnx-model)
        - [5.2 Create config.pbtxt](#52-create-configpbtxt)
    - [Step 6: Launch Triton Inference Server](#step-6-launch-triton-inference-server)
    - [Step 7: Manageable Service with systemd](#step-7-manageable-service-with-systemd)
        - [7.1 Create the systemd Service File](#71-create-the-systemd-service-file)
        - [7.2 Install and Manage the Service](#72-install-and-manage-the-service)
        - [7.3 Controlling and Viewing Logs](#73-controlling-and-viewing-logs)

---

## Installation

This section guides you through setting up the Triton Inference Server with Jina Embeddings v3, optimized for NVIDIA GPUs using TensorRT.

---

### Step 1: System Prerequisites

Ensure your host machine is equipped with the necessary software to interact with NVIDIA GPUs inside Docker containers.

#### 1.1 Install Docker

Install Docker to manage containers for running the Triton Inference Server.

```bash
sudo apt-get update
sudo apt-get install -y docker.io

# Add your user to the docker group to run Docker without sudo
sudo usermod -aG docker ${USER}

echo "----------------------------------------------------"
echo "-> You will need to log out and log back in for docker group changes to apply."
echo "-> Or, you can start a new shell with: newgrp docker"
echo "----------------------------------------------------"
```

**Note:** Log out and back in, or run `newgrp docker` to apply the group changes.

#### 1.2 Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables Docker containers to leverage NVIDIA GPUs.

```bash
# Add the NVIDIA repository and key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime and restart the Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### Step 2: NVIDIA NGC Authentication

Authenticate with the NVIDIA NGC catalog to access the Triton Server container image.

#### 2.1 Get Your NGC API Key

1. Visit **[https://ngc.nvidia.com](https://ngc.nvidia.com)**.
2. Sign in or create a free account.
3. Click your username in the top-right corner and select **Setup**.
4. Click **Get API Key** and then **Generate API Key**.
5. Copy the long alphanumeric API key and store it securely. **Note:** You cannot retrieve it again after closing the page.

#### 2.2 Log in via Docker

Use the API key to authenticate with the NVIDIA Container Registry (`nvcr.io`).

```bash
docker login nvcr.io
```

- **Username:** Enter `$oauthtoken` (literal string).
- **Password:** Paste your NGC API Key (it will not be visible when typed/pasted).

Upon success, you will see a `Login Succeeded` message.

---

### Step 3: Download Jina Embeddings v3 Model

Download the Jina Embeddings v3 model weights from Hugging Face and set up directories for the model data.

```bash
# Create directories on the host machine
mkdir ~/triton_repo

# Install the Hugging Face command-line tool
pip install huggingface_hub

# Log in with your Hugging Face token (it will prompt you to paste it)
huggingface-cli login

# Download the model files into the data directory
huggingface-cli download jinaai/jina-embeddings-v3 \
  --repo-type model \
  --include "onnx/model.onnx*" \
  --local-dir ~/jina-v3-onnx
```

---

### Step 4: Pull Triton Docker Image

Download the Triton Inference Server Docker image from NVIDIA's container registry.

```bash
docker pull nvcr.io/nvidia/tritonserver:25.05-py3
```

---

### Step 5: Prepare Triton Model Repository

Triton requires a specific directory structure for the model repository:

```
~/triton_repo/
└── jina_embeddings_v3/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

#### 5.1 Move the ONNX Model

Copy the downloaded ONNX model files to the Triton model repository.

```bash
mkdir -p ~/triton_repo/jina_embeddings_v3/1
cp ~/jina-v3-onnx/onnx/model.onnx ~/triton_repo/jina_embeddings_v3/1/
cp ~/jina-v3-onnx/onnx/model.onnx_data ~/triton_repo/jina_embeddings_v3/1/
```

#### 5.2 Create config.pbtxt

Create the configuration file for the Jina Embeddings v3 model.

```bash
nano ~/triton_repo/jina_embeddings_v3/config.pbtxt
```

Copy and paste the following content, then save and exit (`Ctrl+X`, `Y`, `Enter`):

```
name: "jina_embeddings_v3"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "task_id"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }
]

output [
  {
    name: "text_embeds"
    data_type: TYPE_FP32
    dims: [-1, 1024 ]
  }
]
```

---

### Step 6: Launch Triton Inference Server

Run the Triton Inference Server in a Docker container, mapping the model repository and exposing necessary ports.

```bash
docker run -d --gpus=all --rm -it \
  --name triton_server \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:25.05-py3 \
  tritonserver --model-repository=/models
```

---

### Step 7: Manageable Service with systemd

Set up a `systemd` service to manage the Triton Inference Server, ensuring it runs reliably and restarts automatically if needed.

#### 7.1 Create the systemd Service File

Create a `systemd` service file to manage the Triton server.

```bash
sudo nano /etc/systemd/system/jinav3embedder.service
```

Copy and paste the following template, replacing placeholders (`<your_username>`, `<full_path_to_your_project>`, `<your_conda_install_dir>`, `<your_conda_env_name>`) with your actual values.

```ini
[Unit]
Description=Jina V3 Embedder Service
After=network.target docker.service

[Service]
User=<your_username>
Group=<your_username>
WorkingDirectory=<full_path_to_your_project>
Environment="PATH=/home/<your_username>/<your_conda_install_dir>/envs/<your_conda_env_name>/bin:/usr/bin:/bin"
ExecStart=/bin/bash <full_path_to_your_project>/run.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Finding Placeholders:**
- `<your_username>`: Run `whoami` to get your username.
- `<full_path_to_your_project>`: Run `pwd` in your project directory (e.g., `~/Triton-Jina-v3-TRT`).
- `<your_conda_install_dir>`: Typically `anaconda3` or `miniconda3` in your home directory.
- `<your_conda_env_name>`: The name of your conda environment (e.g., `triton`).

**Example Filled-Out File:**

For a user `ansary`, project directory `/home/ansary/Triton-Jina-v3-TRT`, and conda environment `triton`, the file would look like:

```ini
[Unit]
Description=Jina V3 Embedder Service
After=network.target docker.service

[Service]
User=ansary
Group=ansary
WorkingDirectory=/home/ansary/Triton-Jina-v3-TRT
Environment="PATH=/home/ansary/miniconda3/envs/triton/bin:/usr/bin:/bin"
ExecStart=/bin/bash /home/ansary/Triton-Jina-v3-TRT/run.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save and exit (`Ctrl+X`, `Y`, `Enter`).

#### 7.2 Install and Manage the Service

Use `systemctl` to manage the `jinav3embedder` service.

1. **Reload the systemd daemon:**
   ```bash
   sudo systemctl daemon-reload
   ```

2. **Enable the service to start on boot:**
   ```bash
   sudo systemctl enable jinav3embedder.service
   ```

3. **Start the service immediately:**
   ```bash
   sudo systemctl start jinav3embedder.service
   ```

#### 7.3 Controlling and Viewing Logs

Manage the service and monitor its logs with the following commands:

- **Check the service status:**
  ```bash
  sudo systemctl status jinav3embedder.service
  ```
  Look for a green `active (running)` status to confirm the service is running correctly.

- **Stop the service:**
  ```bash
  sudo systemctl stop jinav3embedder.service
  ```

- **Restart the service:**
  ```bash
  sudo systemctl restart jinav3embedder.service
  ```

- **View live logs for debugging:**
  ```bash
  sudo journalctl -u jinav3embedder.service -f
  ```
  Press `Ctrl+C` to exit the live log view.

---

This guide provides a comprehensive setup for running the Triton Inference Server with Jina Embeddings v3. Save this markdown file and execute the steps in order to deploy the service successfully.