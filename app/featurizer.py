from transformers import AutoTokenizer, AutoModel
import torch
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
import numpy as np
from uuid import uuid4
from loguru import logger
from configs import MONGO_CONNECTION_URL, QDRANT_URL, QDRANT_API_KEY


class Featurizer:

    try:

        # MongoDB Setup
        mongo_client = MongoClient(MONGO_CONNECTION_URL)
        db = mongo_client["rag"]

        raw_data_collection = db["rag_raw_data"]
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
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        def preprocess_text(self, text):
            """Preprocess text data for featurization."""
            return text.lower().strip()

        def generate_embeddings(self, text):
            """Generate embeddings using a transformer model."""
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean Pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            return embeddings.numpy()

        def featurize_and_store(self):
            """Featurize raw data and store it in MongoDB and Qdrant."""
            raw_docs = self.raw_data_collection.find({"platform": "youtube"})

            for doc in raw_docs:

                # print(doc)

                # break

                # Check if document already exists in the featurized collection
                if self.featurized_collection.find_one({"id": doc["_id"]}):
                    logger.info(
                        f"Document already exists in the database: {doc['_id']}"
                    )
                    continue

                # Preprocess text and generate embeddings
                text = self.preprocess_text(doc["content"])

                embeddings = self.generate_embeddings(text)

                # Store in MongoDB
                self.featurized_collection.insert_one(
                    {
                        "id": doc["_id"],
                        "text": text,
                        "embedding": embeddings.tolist(),
                        "platform": doc["platform"],
                        "url": doc["url"],
                    }
                )

                logger.info("Stored in MongoDB")

                # check if collection rag_vectors exists in Qdrant else create
                if not self.qdrant_client.collection_exists(
                    collection_name="rag_vectors"
                ):
                    self.qdrant_client.create_collection(
                        collection_name="rag_vectors", vectors_config=self.vector_params
                    )

                # Store in Qdrant
                self.qdrant_client.upsert(
                    collection_name="rag_vectors",
                    points=[
                        PointStruct(
                            id=str(uuid4()), vector=embeddings, payload={"text": text}
                        )
                    ],
                )

            logger.info("Featurization complete and data stored.")

    except Exception as e:
        logger.error(f"Error featurizing data: {e}")


if __name__ == "__main__":

    logger.info("Featurizing data...")

    featurizer = Featurizer()

    # Featurize data
    featurizer.featurize_and_store()

    logger.info("Featurization complete.")
