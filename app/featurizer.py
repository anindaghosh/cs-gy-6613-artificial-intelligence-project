from transformers import AutoTokenizer, AutoModel
import torch
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
import numpy as np
from uuid import uuid4
from configs import MONGO_CONNECTION_URL, QDRANT_URL, QDRANT_API_KEY

# MongoDB Setup
mongo_client = MongoClient(MONGO_CONNECTION_URL)
db = mongo_client["rag"]

raw_data_collection = db["pages"]
featurized_collection = db["featurized_data"]

# Qdrant Setup
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
VECTOR_DIM = 384

# Define the vector configuration
vector_params = VectorParams(
    size=VECTOR_DIM,  # Dimensionality of the vectors
    distance="Cosine",  # Similarity metric
)

# Model Setup
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def preprocess_text(text):
    """Preprocess text data for featurization."""
    return text.lower().strip()


def generate_embeddings(text):
    """Generate embeddings using a transformer model."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean Pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()


def featurize_and_store():
    """Featurize raw data and store it in MongoDB and Qdrant."""
    raw_docs = raw_data_collection.find()

    for doc in raw_docs:
        # Preprocess text and generate embeddings
        text = preprocess_text(doc["content"])

        print(text)

        embeddings = generate_embeddings(text)

        print(embeddings)

        # Store in MongoDB
        featurized_collection.insert_one(
            {"id": doc["_id"], "text": text, "embedding": embeddings.tolist()}
        )

        print("Stored in MongoDB")

        # check if collection rag_vectors exists in Qdrant else create
        if not qdrant_client.collection_exists(collection_name="rag_vectors"):
            qdrant_client.create_collection(
                collection_name="rag_vectors", vectors_config=vector_params
            )

        # Store in Qdrant
        qdrant_client.upsert(
            collection_name="rag_vectors",
            points=[
                PointStruct(id=str(uuid4()), vector=embeddings, payload={"text": text})
            ],
        )
    print("Featurization complete and data stored.")


# Featurize data
featurize_and_store()
