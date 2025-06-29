import os
import asyncio
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
# Use environment variables for production-readiness
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8000")
MODEL_NAME = "jina_embeddings_v3"
MODEL_VERSION = "1"
TOKENIZER_NAME = "jinaai/jina-embeddings-v3"

# --- Global objects to be initialized on startup ---
# This avoids reloading the model/client on every request
triton_client: InferenceServerClient
tokenizer: AutoTokenizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A context manager to handle the application's startup and shutdown logic.
    - On startup, it initializes the Triton client and the tokenizer.
    - On shutdown, it gracefully closes the Triton client.
    """
    global triton_client, tokenizer
    print("--- Client App Starting Up ---")
    
    # Initialize Triton Inference Server client
    # The 'async' client is part of the 'tritonclient.http' package
    triton_client = InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
    
    # Initialize the Hugging Face tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer and Triton client initialized.")
    
    yield  # The application runs here
    
    print("--- Client App Shutting Down ---")
    # Cleanly close the client connection
    await triton_client.close()
    print("Triton client closed.")

# Initialize FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

# --- Pydantic Models for API Data Validation ---

class EmbeddingRequest(BaseModel):
    """Request model for the embeddings endpoint."""
    texts: List[str] = Field(..., description="A list of texts to be embedded.", example=["Hello world!", "This is a test."])
    task_id: int = Field(0, description="Task ID for the model (default is 0).")

class Embedding(BaseModel):
    """Represents a single embedding vector."""
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    """Response model for the embeddings endpoint."""
    data: List[Embedding]
    model: str = MODEL_NAME

# --- API Endpoint ---

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Takes a list of texts and returns their corresponding embeddings.
    """
    global triton_client, tokenizer

    try:
        # 1. Tokenize the input texts
        # padding=True ensures all sequences in a batch have the same length.
        # truncation=True cuts sequences longer than the model's max length.
        # return_tensors="np" returns NumPy arrays, which Triton client needs.
        tokenized_inputs = tokenizer(
            request.texts, padding=True, truncation=True, return_tensors="np"
        )
        
        batch_size = len(request.texts)

        # 2. Prepare the input tensors for Triton
        inputs = []
        inputs.append(
            InferInput("input_ids", tokenized_inputs['input_ids'].shape, "INT64")
        )
        inputs.append(
            InferInput("attention_mask", tokenized_inputs['attention_mask'].shape, "INT64")
        )
        # The task_id is a single value per item in the batch
        task_id_array = np.full((batch_size, 1), request.task_id, dtype=np.int64)
        inputs.append(
            InferInput("task_id", task_id_array.shape, "INT64")
        )

        # Set the data for the input tensors
        inputs[0].set_data_from_numpy(tokenized_inputs['input_ids'])
        inputs[1].set_data_from_numpy(tokenized_inputs['attention_mask'])
        inputs[2].set_data_from_numpy(task_id_array)

        # 3. Define the output tensor we want from Triton
        outputs = [InferInput("text_embeds", [batch_size, 1024], "FP32")]
        
        # 4. Send the async inference request to Triton
        print(f"Sending request for batch of {batch_size} to Triton...")
        response: InferResult = await triton_client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inputs=inputs,
            outputs=outputs,
        )

        # 5. Process the response
        raw_embeddings = response.as_numpy("text_embeds")
        
        # Format the output to match the Pydantic response model
        response_data = [
            Embedding(embedding=embedding.tolist(), index=i)
            for i, embedding in enumerate(raw_embeddings)
        ]

        return EmbeddingResponse(data=response_data)

    except InferenceServerException as e:
        # Handle errors from the Triton server gracefully
        print(f"Error from Triton Server: {e}")
        raise HTTPException(
            status_code=503, detail=f"Triton server failed: {e}"
        )
    except Exception as e:
        # Handle other potential errors
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

# A simple root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Jina Embeddings v3 Triton Client is running"}

# To run the app directly for development
if __name__ == "__main__":
    uvicorn.run("client:app", host="0.0.0.0", port=8080, reload=True)