import argparse

import requests
from loguru import logger
from qdrant_client import QdrantClient

from app.configs import OLLAMA_MODEL, OLLAMA_URL, QDRANT_API_KEY, QDRANT_URL
from app.featurizer import Featurizer

COLLECTION_NAME = "rag_vectors"

PROMPT_TEMPLATE = """Answer the question about ROS2 robotics using the context below. \
If the context does not contain the answer, say so instead of guessing.

Context:
{context}

Question: {question}"""


class RagPipeline:

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # reuse the featurizer so queries are embedded exactly like the stored chunks
        self.featurizer = Featurizer()

    def retrieve(self, question: str):
        """Embed the question and return the closest chunks from Qdrant."""
        query_vector = self.featurizer.generate_embeddings(
            self.featurizer.preprocess_text(question)
        )
        result = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=self.top_k,
            with_payload=True,
        )
        return result.points

    def generate(self, question: str, context: str) -> str:
        """Ask the fine-tuned model served by Ollama, grounded in the retrieved context."""
        res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": PROMPT_TEMPLATE.format(context=context, question=question),
                "stream": False,
            },
            timeout=300,
        )
        res.raise_for_status()
        return res.json()["response"].strip()

    def answer(self, question: str):
        hits = self.retrieve(question)
        context = "\n\n".join(hit.payload["text"] for hit in hits)
        logger.info(f"Retrieved {len(hits)} chunks for: {question}")

        answer = self.generate(question, context)
        # chunks embedded before source metadata was stored have no url
        sources = sorted({hit.payload["url"] for hit in hits if hit.payload.get("url")})
        return answer, sources


def main():
    parser = argparse.ArgumentParser(description="Ask the ROS2 RAG assistant a question.")
    parser.add_argument("question", help="question about ROS2, Nav2, MoveIt2 or Gazebo")
    parser.add_argument("--top-k", type=int, default=5, help="number of chunks to retrieve")
    args = parser.parse_args()

    answer, sources = RagPipeline(top_k=args.top_k).answer(args.question)

    print(answer)
    if sources:
        print("\nSources:")
        for url in sources:
            print(f"- {url}")


if __name__ == "__main__":
    main()
