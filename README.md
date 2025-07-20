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

### Step 3: Jina Embeddings v3 Model

* Clone the repository 

```bash
git clone https://github.com/mnansary/Triton-Jina-v3-TRT.git
```

* Download the Jina Embeddings v3 model weights from Hugging Face and set up directories for the model data.

```bash
# Create directories on the host machine
mkdir ~/jinav3_repo
# copy the taskid harcoder script
cp hardcode_taskid.py ~/jinav3_repo/
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

### Step 4: Inspect The model with TensortRT and Polygraphy and Convert

Download the TensorRT Docker image from NVIDIA's container registry.

```bash
docker pull nvcr.io/nvidia/tensorrt:25.05-py3
docker run --gpus all -it --rm -v ~/jinav3_repo:/models nvcr.io/nvidia/tensorrt:25.05-py3
# inside docker 
python3 -m pip install onnx onnx_graphsurgeon onnxruntime
polygraphy inspect model /models/onnx/model_fp16.onnx --mode=onnx
```

* passage model conversion

```bash
# In the TensorRT container
# Step 1.1: Create the passage ONNX model
python3 /models/hardcode_taskid.py \
    --input /models/onnx/model_fp16.onnx \
    --output /models/jina_passage.onnx \
    --task_id 1

# Step 1.2: Build the passage TensorRT engine
polygraphy run /models/jina_passage.onnx \
    --trt \
    --save-engine /models/jina_passage.engine \
    --trt-min-shapes input_ids:[1,1] attention_mask:[1,1] \
    --trt-opt-shapes input_ids:[1,512] attention_mask:[1,512] \
    --trt-max-shapes input_ids:[1,8192] attention_mask:[1,8192] \
    --verbose
```

* query model creation 

```bash 

# In the TensorRT container
# Step 1.1: Create the query ONNX model
python3 /models/hardcode_taskid.py \
    --input /models/onnx/model_fp16.onnx \
    --output /models/jina_query.onnx \
    --task_id 0

# Step 1.2: Build the query TensorRT engine with a DYNAMIC BATCH profile
polygraphy run /models/jina_query.onnx \
    --trt \
    --save-engine /models/jina_query.engine \
    --trt-min-shapes input_ids:[1,1] attention_mask:[1,1] \
    --trt-opt-shapes input_ids:[8,128] attention_mask:[8,128] \
    --trt-max-shapes input_ids:[8,512] attention_mask:[8,512] \
    --verbose
```

* exit docker
```bash
exit
```

### Step-5 Inference With Triton

```bash
# Create the main model repository directory
mkdir -p ~/triton_repo

# Create subdirectories for each of your two models
mkdir -p ~/triton_repo/jina_query/1
mkdir -p ~/triton_repo/jina_passage/1


# On your host machine
# Move the query engine
mv ~/jinav3_repo/jina_query.engine ~/triton_repo/jina_query/1/model.plan
cp configs/jina_query_config.pbtxt ~/triton_repo/jina_query/config.pbtxt

# Move the passage engine
mv ~/jinav3_repo/jina_passage.engine ~/triton_repo/jina_passage/1/model.plan
cp configs/jina_passage_config.pbtxt ~/triton_repo/jina_passage/config.pbtxt
```

* after this the triton_repo should have the following structre: 

```bash
triton_repo/
├── jina_passage/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
└── jina_query/
    ├── 1/
    │   └── model.plan
    └── config.pbtxt
```

* pull the trition server 

```bash
docker pull nvcr.io/nvidia/tritonserver:25.05-py3
```

* run

```bash
docker run --gpus all --rm -d --name jinatriton\
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:25.05-py3 \
  tritonserver --model-repository=/models
```

### Step-6: Service verification

```bash
conda create -n jinatriton python=3.10
conda activate jinatriton
# Install the Triton client, ONNX Runtime for GPU, transformers, and torch
pip install tritonclient[http] onnxruntime-gpu transformers torch
# Jina's ONNX export uses an older numpy, so let's be safe
pip install 'numpy<2'
```

---