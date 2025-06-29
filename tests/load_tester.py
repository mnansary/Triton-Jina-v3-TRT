import asyncio
import aiohttp
import time
import random
import argparse
from tqdm.asyncio import tqdm

# --- Configuration ---
DEFAULT_URL = "http://localhost:24434/v1/embeddings"
HEADERS = {"Content-Type": "application/json"}

# --- NEW: Define a realistic distribution of text lengths for the load test ---
# Format: (Category Name, Min Words, Max Words, Weight/Probability)
# This simulates a workload with many short queries and fewer long documents.
LENGTH_DISTRIBUTION = [
    ("short", 10, 50, 0.60),
    ("medium", 100, 300, 0.25),
    ("long", 500, 1000, 0.10),
    ("very_long", 2000, 3000, 0.05), # Specifically to test the 3k token requirement
]

# Extract names and weights for random.choices
LENGTH_CATEGORIES = [item[0] for item in LENGTH_DISTRIBUTION]
LENGTH_WEIGHTS = [item[3] for item in LENGTH_DISTRIBUTION]
LENGTH_MAP = {item[0]: (item[1], item[2]) for item in LENGTH_DISTRIBUTION}


def generate_text(word_count: int) -> str:
    """Generates a string of dummy text with a specific word count."""
    # For a load test, the content is irrelevant, only the length (and token count).
    # Using a simple repeated word is highly efficient.
    return "test " * word_count


async def make_request(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore):
    """
    Makes a single asynchronous request with dynamically generated text.
    """
    async with semaphore:
        try:
            # --- NEW: Generate varied text for each request ---
            # 1. Choose a length category based on the defined weights
            chosen_category = random.choices(LENGTH_CATEGORIES, LENGTH_WEIGHTS)[0]
            
            # 2. Get the min/max word count for that category
            min_words, max_words = LENGTH_MAP[chosen_category]
            
            # 3. Generate a random word count in that range
            word_count = random.randint(min_words, max_words)
            
            # 4. Create the text payload
            text_to_send = generate_text(word_count)

            payload = {
                "inputs": [
                    {
                        "text": text_to_send,
                        "task_id": random.randint(0, 4)
                    }
                ]
            }
            
            async with session.post(url, json=payload, headers=HEADERS) as response:
                if response.status == 200:
                    await response.json()
                    return True
                else:
                    return False
        except aiohttp.ClientError:
            return False

async def main():
    """Main function to orchestrate the load test."""
    
    parser = argparse.ArgumentParser(description="Advanced Asynchronous Load Tester for Jina V3 Embedding Service.")
    parser.add_argument("-r", "--requests", type=int, default=1000, help="Total number of requests to send.")
    parser.add_argument("-c", "--concurrency", type=int, default=100, help="Number of simultaneous requests.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="URL of the embedding service.")
    args = parser.parse_args()

    print("--- Jina V3 Embedder Advanced Load Test ---")
    print(f"Target URL: {args.url}")
    print(f"Total Requests: {args.requests}")
    print(f"Concurrency Level: {args.concurrency}")
    print(f"Text Length Distribution: {[(item[0], f'{item[3]*100:.0f}%') for item in LENGTH_DISTRIBUTION]}")
    print("---------------------------------------------")

    semaphore = asyncio.Semaphore(args.concurrency)
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, args.url, semaphore) for _ in range(args.requests)]
        results = await tqdm.gather(*tasks, desc="Sending Varied-Length Requests")

    end_time = time.time()
    
    total_time = end_time - start_time
    success_count = sum(1 for r in results if r is True)
    failure_count = len(results) - success_count
    rps = success_count / total_time if total_time > 0 else 0

    print("\n--- Load Test Summary ---")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests: {failure_count}")
    print(f"Requests Per Second (RPS): {rps:.2f}")
    print("-------------------------")

if __name__ == "__main__":
    asyncio.run(main())