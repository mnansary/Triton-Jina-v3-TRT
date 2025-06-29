# Triton-Jina-v3-TRT

- [**Installation**](#installation)
    -   [**Step 1: System Prerequisites**](#step-1-system-prerequisites)
        -   [1.1 Install Docker](#11-install-docker)
        -   [1.2 Install NVIDIA Container Toolkit](#12-install-nvidia-container-toolkit)
    -   [**Step 2: NVIDIA NGC Authentication**](#step-2-nvidia-ngc-authentication)
        -   [2.1 Get Your NGC API Key](#21-get-your-ngc-api-key)
        -   [2.2 Log in via Docker](#22-log-in-via-docker)
    -   [**Step 3: Download Triton Server Image**](#step-3-download-triton-server-image)


# Installation

---

## Step 1: System Prerequisites

This section covers the base software required on your host machine to interact with NVIDIA GPUs inside Docker containers.

### 1.1 Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io

# Add your user to the docker group to run docker without sudo
# NOTE: You must log out and log back in for this change to take effect.
sudo usermod -aG docker ${USER}

echo "----------------------------------------------------"
echo "-> You will need to log out and log back in for docker group changes to apply."
echo "-> Or, you can start a new shell with: newgrp docker"
echo "----------------------------------------------------"
```

### 1.2 Install NVIDIA Container Toolkit

This toolkit allows Docker containers to access your NVIDIA GPU.

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

## Step 2: NVIDIA NGC Authentication

To download the pre-built Triton container, you need to authenticate with the NVIDIA NGC catalog.

### 2.1 Get Your NGC API Key

1.  Go to the NVIDIA NGC website: **[https://ngc.nvidia.com](https://ngc.nvidia.com)**
2.  Sign in or create a free account.
3.  In the top-right corner, click your user name and select **"Setup"**.
4.  On the Setup page, click **"Get API Key"** and then **"Generate API Key"**.
5.  **IMPORTANT:** A long alphanumeric string will be displayed. This is your API key. Copy this key immediately and save it somewhere safe. You will not be able to see it again.

### 2.2 Log in via Docker

Use the API key to log in to the NVIDIA Container Registry (`nvcr.io`).

```bash
docker login nvcr.io
```

The command will prompt you for a `Username` and a `Password`:

-   **Username:** Enter the literal string `$oauthtoken`
-   **Password:** Paste the **NGC API Key** you just generated. (Note: The password will be invisible as you type/paste it).

You should see a `Login Succeeded` message.

---


## Step 3: Download the jina-v3-embeddings  

We will download the model weights from Hugging Face and create directories to store model data and the final Triton repository.

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


## Step-5: Pull triton docker 
```bash
docker pull nvcr.io/nvidia/tritonserver:25.05-py3
```

## Step 6: Prepare Triton Model Repository
Triton expects a specific structure like this:

```bash
Edit
~/triton_repo/
└── jina_embeddings_v3/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

### 6.1 Move the ONNX model
```bash
mkdir -p ~/triton_repo/jina_embeddings_v3/1
cp ~/jina-v3-onnx/onnx/model.onnx ~/triton_repo/jina_embeddings_v3/1/
cp ~/jina-v3-onnx/onnx/model.onnx_data ~/triton_repo/jina_embeddings_v3/1/
```
### 4.2 Create config.pbtxt
Create the config file at: **~/triton_repo/jina_embeddings_v3/config.pbtxt**

```bash
nano ~/triton_repo/jina_embeddings_v3/config.pbtxt
```

copy and paste and save the following

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
    dims: [-1,1024 ]  # Adjust based on model output
  }
]
```
# Step 7: Launch Triton Inference Server
```bash
docker run -d --gpus=all --rm -it \
  --name triton_server \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:25.05-py3 \
  tritonserver --model-repository=/models
```

