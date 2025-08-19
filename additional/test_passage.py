import os
import time
import psutil
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer

TRITON_URL = "localhost:4000"
MODEL_NAME = "jina_passage"
MODEL_VERSION = "1"   # or "" for latest

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")

# Example texts
texts = [
    "Dhaka is the capital of Bangladesh.",
    "The Padma river flows across Bangladesh."
]

# Tokenize
encoded = tokenizer(
    texts,
    padding="longest",
    truncation=True,
    return_tensors="np"
)

input_ids = encoded["input_ids"].astype(np.int64)
attention_mask = encoded["attention_mask"].astype(np.int64)

# Triton client
client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)

# Create Triton inputs
inputs = []
inputs.append(httpclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)))
inputs[-1].set_data_from_numpy(input_ids)

inputs.append(httpclient.InferInput("attention_mask", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)))
inputs[-1].set_data_from_numpy(attention_mask)

# Request outputs
outputs = []
outputs.append(httpclient.InferRequestedOutput("text_embeds"))
outputs.append(httpclient.InferRequestedOutput("13049"))  # pooled embeddings

# ---------------- CPU + RAM MONITOR ----------------
proc = psutil.Process(os.getpid())

cpu_before = proc.cpu_percent(interval=None)
mem_before = proc.memory_info().rss / (1024 * 1024)

start_time = time.time()

# Run inference
response = client.infer(model_name=MODEL_NAME, model_version=MODEL_VERSION, inputs=inputs, outputs=outputs)

end_time = time.time()

cpu_after = proc.cpu_percent(interval=None)  # %
mem_after = proc.memory_info().rss / (1024 * 1024)  # MB

# ----------------------------------------------------

# Extract numpy results
text_embeds = response.as_numpy("text_embeds")
pooled_embeds = response.as_numpy("13049")

print("==== Input Texts ====")
for t in texts:
    print("  ", t)

print("\n==== Output Shapes ====")
print("text_embeds:", text_embeds.shape)      # (batch, seq_len, 1024)
print("pooled_embeds:", pooled_embeds.shape)  # (batch, 1024)

print("\n==== First row of pooled embeddings (truncated) ====")
print(pooled_embeds[0][:10])  # show first 10 dims

print("\n==== Resource Usage ====")
print(f"Execution Time: {end_time - start_time:.3f} sec")
print(f"CPU Usage: {cpu_after:.2f}%")
print(f"Memory Before: {mem_before:.2f} MB")
print(f"Memory After:  {mem_after:.2f} MB")
