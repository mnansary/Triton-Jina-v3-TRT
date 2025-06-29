import os
from contextlib import asynccontextmanager
from typing import List
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from tritonclient.http import InferenceServerClient, InferInput, InferResult
from tritonclient.utils import InferenceServerException

# --- Configuration ---
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8000")
MODEL_NAME = "jina_embeddings_v3"
MODEL_VERSION = "1"
TOKENIZER_NAME = "jinaai/jina-embeddings-v3"

# --- Global objects to be initialized on startup ---
triton_client: InferenceServerClient
tokenizer: AutoTokenizer

# Mean pooling function
def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, model_output.shape)
    sum_embeddings = np.sum(model_output * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

@asynccontextmanager
async def lifespan(app: FastAPI):
    global triton_client, tokenizer
    print("--- Client App Starting Up ---")
    triton_client = InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer and Triton client initialized.")
    yield
    print("--- Client App Shutting Down ---")
    triton_client.close()
    print("Triton client closed.")

app = FastAPI(lifespan=lifespan)

class EmbeddingInputItem(BaseModel):
    text: str
    task_id: int

class EmbeddingRequest(BaseModel):
    inputs: List[EmbeddingInputItem]

class Embedding(BaseModel):
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    data: List[Embedding]
    model: str = MODEL_NAME

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    global triton_client, tokenizer

    try:
        texts = [item.text for item in request.inputs]
        task_ids = [item.task_id for item in request.inputs]
        
        if not texts:
            return EmbeddingResponse(data=[])

        print("\n--- [1] Processing Request ---") # <-- ADDED

        # 2. Tokenize the input texts
        tokenized_inputs = tokenizer(
            texts, padding=True, truncation=True, return_tensors="np"
        )
        # <-- ADDED: Print shapes after tokenization
        print(f"Shape of tokenized input_ids: {tokenized_inputs['input_ids'].shape}")
        print(f"Shape of tokenized attention_mask: {tokenized_inputs['attention_mask'].shape}")
        
        batch_size = len(texts)
        task_id_array = np.array([[task] for task in task_ids], dtype=np.int64)
        print(f"Shape of task_id_array: {task_id_array.shape}") # <-- ADDED

        # 3. Prepare the input tensors for Triton
        inputs = []
        inputs.append(InferInput("input_ids", tokenized_inputs['input_ids'].shape, "INT64"))
        inputs.append(InferInput("attention_mask", tokenized_inputs['attention_mask'].shape, "INT64"))
        inputs.append(InferInput("task_id", task_id_array.shape, "INT64"))

        inputs[0].set_data_from_numpy(tokenized_inputs['input_ids'])
        inputs[1].set_data_from_numpy(tokenized_inputs['attention_mask'])
        inputs[2].set_data_from_numpy(task_id_array)
        
        # 5. Send the inference request
        print("\n--- [2] Sending Request to Triton ---") # <-- ADDED
        response: InferResult = triton_client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inputs=inputs
        )

        # 6. Process the response
        raw_embeddings = response.as_numpy("text_embeds")
        print("\n--- [3] Processing Triton Response ---") # <-- ADDED
        # <-- ADDED: THIS IS THE MOST IMPORTANT SHAPE TO VERIFY
        print(f"Shape of raw_embeddings FROM TRITON: {raw_embeddings.shape}") 
        
        embeddings = mean_pooling(raw_embeddings, tokenized_inputs['attention_mask'])
        # <-- ADDED: Print shape after pooling to confirm it's 2D
        print(f"Shape of embeddings AFTER MEAN POOLING: {embeddings.shape}") 
        
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
        
        response_data = [
            Embedding(embedding=embedding.tolist(), index=i)
            for i, embedding in enumerate(embeddings)
        ]

        print("\n--- [4] Request Complete ---\n") # <-- ADDED
        return EmbeddingResponse(data=response_data, model=MODEL_NAME)

    except InferenceServerException as e:
        print(f"Error from Triton Server: {e}")
        raise HTTPException(status_code=503, detail=f"Triton server failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Jina Embeddings v3 Triton Client is running"}

if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8080, reload=True)