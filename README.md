# Triton-Jina-v3-TRT

## Installation

This section guides you through setting up the Triton Inference Server with Jina Embeddings v3, optimized for NVIDIA GPUs using the TensorRT runtime.

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
mkdir ~/jinav3_repo

# Install the Hugging Face command-line tool
pip install huggingface_hub

# Log in with your Hugging Face token (it will prompt you to paste it)
huggingface-cli login

# Download the model files into the data directory
huggingface-cli download jinaai/jina-embeddings-v3 \
  --repo-type model \
  --include "onnx/model*" \
  --local-dir ~/jinav3_repo
```

---

### Step 4: Pull TensorRT Docker Image

Download the TensorRT Docker image from NVIDIA's container registry.

```bash
docker pull nvcr.io/nvidia/tensorrt:25.05-py3
```

---