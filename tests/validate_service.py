import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig
import requests
import json
import os

# --- Configuration ---
SERVICE_URL = "http://localhost:24434/v1/embeddings"
MODEL_DIR = os.path.expanduser('~/jina-v3-onnx') 
MODEL_PATH = os.path.join(MODEL_DIR, 'onnx/model.onnx')
TOKENIZER_NAME = 'jinaai/jina-embeddings-v3'
# CORRECTED: Use a similarity threshold for validation, not a strict element-wise check.
SIMILARITY_THRESHOLD = 0.99999 

# --- Helper Functions (No changes needed here) ---

def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def get_local_onnx_embedding(text: str, task_id: int, session: onnxruntime.InferenceSession, tokenizer) -> np.ndarray:
    tokenized_input = tokenizer(text, return_tensors='np')
    inputs = {
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'task_id': np.array([task_id], dtype=np.int64)
    }
    model_output = session.run(None, inputs)[0]
    embedding = mean_pooling(model_output, tokenized_input["attention_mask"])
    embedding = embedding / np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
    return embedding.flatten()

def get_service_embedding(text: str, task_id: int, service_url: str) -> np.ndarray:
    headers = {"Content-Type": "application/json"}
    payload = {"inputs": [{"text": text, "task_id": task_id}]}
    response = requests.post(service_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    data = response.json()
    embedding = data['data'][0]['embedding']
    return np.array(embedding)

# --- Main Validation Logic ---

def main():
    print("--- Starting Final Production Validation Test ---")

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        config = PretrainedConfig.from_pretrained(TOKENIZER_NAME)
        lora_task_list = config.lora_adaptations
        session = onnxruntime.InferenceSession(MODEL_PATH)
        print("Dependencies loaded successfully.")
    except Exception as e:
        print(f"\n❌ FAILED TO INITIALIZE: {e}")
        return
    
    test_cases = [
        {"text": "This is a document for retrieval.", "task_name": "retrieval.passage"},
        {"text": "What is the capital of France?", "task_name": "retrieval.query"},
        {"text": "A long article about machine learning.", "task_name": "separation"},
        {"text": "Search for AI technologies.", "task_name": "text-matching"},
        {"text": "This text belongs to the 'sports' category.", "task_name": "classification"},
    ]

    print("\n--- Running Comparison ---")
    all_passed = True
    for i, case in enumerate(test_cases):
        task_name = case['task_name']
        task_id = lora_task_list.index(task_name)

        print(f"\n--- Test Case {i+1} ---")
        print(f"Text: '{case['text']}' | Task: '{task_name}' (ID: {task_id})")
        
        try:
            local_embedding = get_local_onnx_embedding(text=case['text'], task_id=task_id, session=session, tokenizer=tokenizer)
            service_embedding = get_service_embedding(text=case['text'], task_id=task_id, service_url=SERVICE_URL)
            
            cosine_similarity = np.dot(local_embedding, service_embedding)
            print(f"  -> Comparison:")
            print(f"     Cosine Similarity: {cosine_similarity:.8f}")
            
            # THE FINAL CHECK: Is the similarity above our acceptable threshold?
            if cosine_similarity >= SIMILARITY_THRESHOLD:
                print(f"     Status: ✅ PASSED (Similarity is >= {SIMILARITY_THRESHOLD})")
            else:
                all_passed = False
                print(f"     Status: ❌ FAILED (Similarity is < {SIMILARITY_THRESHOLD})")

        except Exception as e:
            all_passed = False
            print(f"     Status: ❌ FAILED with unexpected error: {e}")

    print("\n--- Validation Summary ---")
    if all_passed:
        print("✅ All test cases passed! Your service is fully validated and ready for production.")
    else:
        print("❌ Some test cases failed. Please review the output above.")


if __name__ == "__main__":
    main()