import requests
import json
import math
from typing import List

# TASK_MAPPING remains the same (zero-indexed)
TASK_MAPPING = {
    'retrieval.query': 0,
    'retrieval.passage': 1,
    'separation': 2,
    'classification': 3,
    'text-matching': 4,
}

class JinaV3ApiEmbeddings:
    """
    LangChain-compatible embedding class that calls a Jina V3 FastAPI service.
    It supports all documented Jina V3 tasks and includes automatic request batching.
    """
    def __init__(self, host: str = "192.168.20.111", port: int = 24434, timeout: int = 60, batch_size: int = 4):
        """
        Initializes the API-based embedder.

        Args:
            host (str): The IP address or hostname of the FastAPI service.
            port (int): The port the service is running on.
            timeout (int): The request timeout in seconds for each batch.
            batch_size (int): The maximum number of texts to send in a single API call.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.batch_size = batch_size
        self.api_url = f"http://{self.host}:{self.port}/v1/embeddings"
        print(f"JinaV3ApiEmbeddings configured for: {self.api_url}")
        print(f"Max batch size set to: {self.batch_size}")

    def _embed_batch(self, texts: List[str], task_id: int) -> List[List[float]]:
        """A private method to handle the API call for a single batch."""
        if not texts:
            return []

        payload_inputs = [{"text": str(text), "task_id": task_id} for text in texts]
        payload = {"inputs": payload_inputs}

        try:
            response = requests.post(
                self.api_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            response_json = response.json()
            # Sort results by original index to ensure order is maintained
            data = sorted(response_json['data'], key=lambda item: item['index'])
            return [item['embedding'] for item in data]
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error connecting to Jina API: {e}")
            print(f"   Response Body: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to Jina API at {self.api_url}. Error: {e}")
            raise

    def embed(self, texts: List[str], task: str) -> List[List[float]]:
        """
        Generic method to embed a list of texts for a specified task, with batching.

        Args:
            texts (List[str]): The list of texts to embed.
            task (str): The embedding task to perform.

        Returns:
            List[List[float]]: A list of embeddings in the same order as the input texts.
        """
        if task not in TASK_MAPPING:
            raise ValueError(f"Invalid task '{task}'. Available tasks are: {list(TASK_MAPPING.keys())}")
        
        task_id = TASK_MAPPING[task]
        all_embeddings = []
        
        # Calculate number of batches needed
        num_texts = len(texts)
        num_batches = math.ceil(num_texts / self.batch_size)

        print(f"-> Embedding {num_texts} texts for task '{task}' in {num_batches} batches...")

        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            print(f"   - Processing batch {batch_num} of {num_batches} (size: {len(batch_texts)})...")
            
            # The API response is already sorted, but the indices are relative to the batch.
            # Since we process batches in order and extend the list, the final order is preserved.
            batch_embeddings = self._embed_batch(batch_texts, task_id)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convenience wrapper for embed(..., task='retrieval.passage'). Inherits batching."""
        return self.embed(texts, task='retrieval.passage')

    def embed_query(self, text: str) -> list[float]:
        """Convenience wrapper for embed(..., task='retrieval.query'). Inherits batching."""
        embeddings = self.embed([text], task='retrieval.query')
        if not embeddings:
            raise ValueError("API returned no embedding for the query.")
        return embeddings[0]

# --- Test code demonstrating the new batching functionality ---
if __name__ == "__main__":
    try:
        # Initialize with the default batch_size=4
        embedder = JinaV3ApiEmbeddings(host="192.168.20.111", port=24434)
        print("-" * 50)

        # Create a list of 7 documents to force batching (4 + 3)
        documents_for_batch_test = [
            "The sun is the star at the center of the Solar System.", # Batch 1
            "Jupiter is the fifth planet from the Sun and the largest in the Solar System.", # Batch 1
            "The Moon is Earth's only natural satellite.", # Batch 1
            "Mars is the fourth planet from the Sun and the second-smallest planet.", # Batch 1
            "The Great Wall of China is a series of fortifications.", # Batch 2
            "The Amazon River is the largest river by discharge volume of water.", # Batch 2
            "An atom is the smallest unit of ordinary matter." # Batch 2
        ]
        
        print(f"### Testing batching with {len(documents_for_batch_test)} documents and a batch size of {embedder.batch_size} ###")
        
        # Use the embed_documents method, which will trigger the batching logic inside embed()
        document_embeddings = embedder.embed_documents(documents_for_batch_test)
        
        print("\n--- Batching Test Results ---")
        print(f"Total documents sent: {len(documents_for_batch_test)}")
        print(f"Total embeddings received: {len(document_embeddings)}")

        # Crucial check: a successful batching operation must return one embedding for each input text
        assert len(documents_for_batch_test) == len(document_embeddings)
        print("✅ Assertion Passed: The number of embeddings received matches the number of documents sent.")

        if document_embeddings:
            print(f"Dimension of embeddings: {len(document_embeddings[0])}")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the batching test: {e}")