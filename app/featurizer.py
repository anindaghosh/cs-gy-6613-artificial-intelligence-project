import torch
from transformers import AutoTokenizer, AutoModel
from clearml import Task
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from uuid import uuid4
from loguru import logger

from app.configs import QDRANT_URL
from app.helpers.mongo_client import get_mongo_client


class Featurizer:

    try:

        # MongoDB Setup
        # mongo_client = MongoClient(MONGO_CONNECTION_URL)
        # db = mongo_client["rag"]

        # raw_data_collection = db["rag_raw_data"]
        # featurized_collection = db["featurized_data"]

        # Qdrant Setup
        # qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        qdrant_client = QdrantClient(url=QDRANT_URL)

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

        def featurize_and_store(self, chunks, featurized_collection):
            """Featurize raw data and store it in MongoDB and Qdrant."""
            # raw_docs = self.raw_data_collection.find({"platform": "youtube"})

            for i, doc in enumerate(chunks):

                # print(doc)

                # # print(doc)

                # # break

                # # Check if document already exists in the featurized collection
                # if self.featurized_collection.find_one({"id": doc["_id"]}):
                #     logger.info(
                #         f"Document already exists in the database: {doc['_id']}"
                #     )
                #     continue

                # Preprocess text and generate embeddings
                text = self.preprocess_text(doc["content"])

                embeddings = self.generate_embeddings(text)

                # Store in MongoDB
                featurized_collection.insert_one(
                    {
                        # "id": doc["_id"],
                        "text": text,
                        "embedding": embeddings.tolist(),
                        "platform": doc["platform"],
                        "url": doc["url"],
                    }
                )

                logger.info(f"Stored document {i+1} in MongoDB")

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

                logger.info(
                    f"Featurization for document {i+1} complete and data stored."
                )

            # logger.info(f"Featurization for document {i+1} complete and data stored.")

    except Exception as e:
        logger.error(f"Error featurizing data: {e}")


def etl_featurize_step():

    task = Task.init(project_name="cs-gy-6613-rag", task_name="etl_featurizer")

    logger.info("Featurizing data...")

    mongo_client = get_mongo_client()
    db = mongo_client["rag"]
    chunk_collection = db["rag_chunked_data"]
    featurized_collection = db["featurized_data"]

    chunks = chunk_collection.find({}).limit(100)

    featurizer = Featurizer()

    # Featurize data
    featurizer.featurize_and_store(
        chunks=chunks, featurized_collection=featurized_collection
    )

    logger.info("Featurization complete.")

    task.close()


if __name__ == "__main__":

    etl_featurize_step()
