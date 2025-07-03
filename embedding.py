import requests
import json
from typing import List, Dict, Any

# This mapping is the critical link between user-friendly task names
# and the integer IDs required by the Triton/FastAPI backend.
# The integer values are now ZERO-INDEXED as requested.
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
    It supports all documented Jina V3 tasks by mapping them to the correct task_id.
    """
    def __init__(self, host: str = "192.168.20.111", port: int = 24434, timeout: int = 60):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.api_url = f"http://{self.host}:{self.port}/v1/embeddings"
        print(f"JinaV3ApiEmbeddings configured to use service at: {self.api_url}")
        print(f"Supported tasks (with zero-indexed IDs): {TASK_MAPPING}")

    def _call_api(self, texts: List[str], task_id: int) -> List[List[float]]:
        """A private method to handle the actual API call."""
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
        Generic method to embed a list of texts for a specified task.

        Args:
            texts (List[str]): The list of texts to embed.
            task (str): The embedding task to perform. Must be one of the keys
                        in TASK_MAPPING.

        Returns:
            List[List[float]]: A list of embeddings.
        """
        if task not in TASK_MAPPING:
            raise ValueError(
                f"Invalid task '{task}'. "
                f"Available tasks are: {list(TASK_MAPPING.keys())}"
            )
        
        print(f"-> Embedding {len(texts)} texts (task: '{task}', id: {TASK_MAPPING[task]})...")
        task_id = TASK_MAPPING[task]
        return self._call_api(texts, task_id)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents (passages) for storage.
        This is a convenience wrapper around embed(..., task='retrieval.passage').
        """
        return self.embed(texts, task='retrieval.passage')

    def embed_query(self, text: str) -> list[float]:
        """
        Embeds a single query for searching.
        This is a convenience wrapper around embed(..., task='retrieval.query').
        """
        embeddings = self.embed([text], task='retrieval.query')
        if not embeddings:
            raise ValueError("API returned no embedding for the query.")
        return embeddings[0]

# --- Comprehensive Example Usage (No changes needed here) ---
if __name__ == "__main__":
    try:
        # 1. Initialize the embedding client
        embedder = JinaV3ApiEmbeddings(host="192.168.20.111", port=24434)
        print("-" * 50)

        # 2. Standard Retrieval Use Case (using convenience methods)
        print("### Testing Standard Retrieval (Asymmetric) ###")
        docs_to_store = [
            "The Eiffel Tower is in Paris, France.",
            "The Colosseum is an ancient amphitheater in Rome, Italy."
        ]
        doc_embeddings = embedder.embed_documents(docs_to_store)
        print(f"Got {len(doc_embeddings)} document embeddings.")

        query = "What is a famous landmark in Italy?"
        query_embedding = embedder.embed_query(query)
        print(f"Got 1 query embedding of dimension {len(query_embedding)}.")
        print("-" * 50)

        # 3. Text Matching / Symmetric Search Use Case
        print("### Testing Text Matching (Symmetric) ###")
        sts_texts = [
            "A man is eating food.",
            "A man is eating a piece of bread." # Similar
        ]
        matching_embeddings = embedder.embed(sts_texts, task='text-matching')
        print(f"Got {len(matching_embeddings)} text-matching embeddings.")
        print("-" * 50)

        # 4. Classification Use Case
        print("### Testing Classification ###")
        class_texts = [
            "This movie was fantastic, I loved it!",
            "I was really bored by the plot."
        ]
        classification_embeddings = embedder.embed(class_texts, task='classification')
        print(f"Got {len(classification_embeddings)} classification embeddings.")
        print("-" * 50)
        
        # 5. Separation (Clustering / Reranking) Use Case
        print("### Testing Separation ###")
        cluster_texts = [
            "Apple", "Microsoft", "Google", # Tech companies
            "Carrot", "Broccoli", "Spinach"  # Vegetables
        ]
        separation_embeddings = embedder.embed(cluster_texts, task='separation')
        print(f"Got {len(separation_embeddings)} separation embeddings.")
        print("-" * 50)

    except Exception as e:
        print(f"\nAn unexpected error occurred during the example run: {e}")