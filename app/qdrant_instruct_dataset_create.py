from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

from configs import QDRANT_API_KEY, QDRANT_URL

# Initialize Qdrant Client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "rag_vectors"

# Initialize embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


# Helper: Generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]


# Retrieve context from Qdrant
def retrieve_context(question, top_k=3):
    question_embedding = get_embedding(question)
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME, query_vector=question_embedding, limit=top_k
    )
    return [hit.payload["text"] for hit in search_result]


# Generate dataset
def create_instruct_dataset(questions):
    dataset = []
    for question in questions:
        # Retrieve context
        contexts = retrieve_context(question, top_k=1)
        if not contexts:
            continue
        context = contexts[0]  # Use the most relevant context

        # Generate answer (use LLM or domain knowledge for better quality)
        answer = f"Extracted answer based on context: {context[:100]}"  # Replace with actual logic

        # Add to dataset
        dataset.append({"question": question, "context": context, "answer": answer})
    return dataset


# Example questions
questions = [
    "What is the purpose of the linux-real-time-kernel-builder?",
    "How do I configure virtualization in ROS2?",
    "What are the key components of the nav2 navigation stack?",
]

# Create instruct dataset
instruct_dataset = create_instruct_dataset(questions)

# Save dataset to CSV
df = pd.DataFrame(instruct_dataset)
df.to_csv("instruct_dataset.csv", index=False)
print("Dataset saved to instruct_dataset.csv")
