import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import tritonclient.http as httpclient
import sys
import os

# --- Configuration ---
ONNX_MODEL_PATH = os.path.expanduser('~/jinav3_repo/onnx/model_fp16.onnx')
TOKENIZER_PATH = 'jinaai/jina-embeddings-v3' 
TRITON_URL = 'localhost:8000'

TASK_IDS = { 'query': np.array(0, dtype=np.int64), 'passage': np.array(1, dtype=np.int64) }

def cosine_similarity(a, b):
    # Ensure inputs are 2D
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    # Calculate similarity row-by-row
    sim = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return sim

def validate_embeddings(name, onnx_embedding, triton_embedding):
    """
    Validates embeddings for a production environment.
    The primary success metric is high cosine similarity.
    """
    print(f"\n----- Validating: {name} -----")
    
    # Calculate cosine similarity for each vector in the batch
    similarities = cosine_similarity(onnx_embedding, triton_embedding)
    min_similarity = np.min(similarities)
    
    # For production, a cosine similarity > 0.999 is considered an excellent match
    is_similar = min_similarity > 0.999
    
    status = "✅ SUCCESS" if is_similar else "❌ FAILURE"
    
    print(f"{status}: Minimum Cosine Similarity is {min_similarity:.6f}.")
    if is_similar:
        print("   - The FP16 TRT engine is functionally identical to the original model.")
    else:
        print(f"   - WARNING: Similarity is below the 0.999 threshold. A review is needed.")

    return is_similar

# (The get_onnx_embedding and get_triton_embedding functions remain the same)
def get_onnx_embedding(session, tokens, task_id):
    inputs = { 'input_ids': tokens['input_ids'].astype(np.int64), 'attention_mask': tokens['attention_mask'].astype(np.int64), 'task_id': task_id }
    return session.run(None, inputs)[1]

def get_triton_embedding(client, model_name, tokens):
    inputs = [ httpclient.InferInput('input_ids', tokens['input_ids'].shape, "INT64"), httpclient.InferInput('attention_mask', tokens['attention_mask'].shape, "INT64") ]
    inputs[0].set_data_from_numpy(tokens['input_ids'])
    inputs[1].set_data_from_numpy(tokens['attention_mask'])
    response = client.infer(model_name=model_name, inputs=inputs)
    return response.as_numpy('13049')


def main():
    print("Initializing models and client for FP16 Production Validation...")
    # (main function logic is the same as the previous full script)
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        assert triton_client.is_server_live(), "Triton server is not live."
    except Exception as e:
        print(f"ERROR: Could not initialize. {e}")
        sys.exit(1)

    all_passed = True
    try:
        passage_text = ["Jina-v3 is a powerful embedding model."]
        passage_tokens = tokenizer(passage_text, return_tensors='np', padding=True)
        onnx_passage_emb = get_onnx_embedding(ort_session, passage_tokens, TASK_IDS['passage'])
        triton_passage_emb = get_triton_embedding(triton_client, 'jina_passage', passage_tokens)
        if not validate_embeddings("Passage (Batch=1, FP16)", onnx_passage_emb, triton_passage_emb):
            all_passed = False

        query1_text = ["what is the best embedding model?"]
        query1_tokens = tokenizer(query1_text, return_tensors='np', padding=True)
        onnx_q1_emb = get_onnx_embedding(ort_session, query1_tokens, TASK_IDS['query'])
        triton_q1_emb = get_triton_embedding(triton_client, 'jina_query', query1_tokens)
        if not validate_embeddings("Query (Batch=1, FP16)", onnx_q1_emb, triton_q1_emb):
            all_passed = False
        
        query8_text = ["what is the best embedding model?"] * 8
        query8_tokens = tokenizer(query8_text, return_tensors='np', padding=True)
        onnx_q8_emb = get_onnx_embedding(ort_session, query8_tokens, TASK_IDS['query'])
        triton_q8_emb = get_triton_embedding(triton_client, 'jina_query', query8_tokens)
        if not validate_embeddings("Query (Batch=8, FP16)", onnx_q8_emb, triton_q8_emb):
            all_passed = False
    
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        all_passed = False

    print("\n\n--- FINAL PRODUCTION VALIDATION ---")
    if all_passed:
        print("🎉🎉🎉 ALL FP16 ENGINES ARE VALIDATED FOR PRODUCTION USE! 🎉🎉🎉")
    else:
        print("🔥🔥🔥 One or more models failed the functional similarity test. 🔥🔥🔥")


if __name__ == "__main__":
    main()