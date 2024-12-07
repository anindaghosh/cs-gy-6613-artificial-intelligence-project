from typing import List, Dict, Any
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMongoLoader
from langchain.schema import Document
import pymongo
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import uuid


class RagPipeline:
    def __init__(
        self,
        mongo_uri: str,
        collection_name: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        """Initialize the RAG pipeline with database connections."""
        self.mongo_client = pymongo.MongoClient(mongo_uri)
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings()
        self.embedding_dim = 768  # Default for many HuggingFace models

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Create collection if it doesn't exist
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim, distance=Distance.COSINE
            ),
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def clean_docs(self, raw_docs: List[Dict[str, Any]]) -> List[Document]:
        """Clean and preprocess raw documents."""
        cleaned_docs = []
        for doc in raw_docs:
            # Extract text and metadata
            text = doc.get("text", "")
            metadata = {
                "source": doc.get("source", ""),
                "timestamp": doc.get("timestamp", ""),
                "doc_id": str(doc.get("_id", "")),
            }

            # Basic cleaning operations
            text = text.strip()
            if text:  # Only include non-empty documents
                cleaned_docs.append(Document(page_content=text, metadata=metadata))

        return cleaned_docs

    def chunk_docs(self, cleaned_docs: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunked_docs = []
        for doc in cleaned_docs:
            chunks = self.text_splitter.split_documents([doc])
            # Preserve original metadata while adding chunk information
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({"chunk_id": i, "total_chunks": len(chunks)})
                chunked_docs.append(chunk)
        return chunked_docs

    def embed_chunks(self, chunked_docs: List[Document]) -> None:
        """Create embeddings for chunks and store in Qdrant."""
        # Process documents in batches
        batch_size = 100
        for i in range(0, len(chunked_docs), batch_size):
            batch = chunked_docs[i : i + batch_size]

            # Generate embeddings for the batch
            texts = [doc.page_content for doc in batch]
            embeddings = self.embeddings.embed_documents(texts)

            # Prepare points for Qdrant
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": doc.page_content, **doc.metadata},
                )
                for doc, embedding in zip(batch, embeddings)
            ]

            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )

    def process_batch(self) -> None:
        """Run the complete batch processing pipeline."""
        # Load raw documents from MongoDB
        loader = PyMongoLoader(
            connection_string=self.mongo_client,
            db_name="your_db_name",
            collection_name=self.collection_name,
        )
        raw_docs = loader.load()

        # Process through pipeline
        cleaned_docs = self.clean_docs(raw_docs)
        chunked_docs = self.chunk_docs(cleaned_docs)
        self.embed_chunks(chunked_docs)


class RetrievalClient:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        embeddings: HuggingFaceEmbeddings,
    ):
        """Initialize retrieval client with Qdrant client."""
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embeddings = embeddings

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query."""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=k
        )

        # Convert results to Documents
        documents = []
        for result in search_results:
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            documents.append(
                Document(page_content=result.payload["text"], metadata=metadata)
            )

        return documents


class InferencePipeline:
    def __init__(self, model_path: str, retrieval_client: RetrievalClient):
        """Initialize inference pipeline with model and retrieval client."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.retrieval_client = retrieval_client

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_answer(self, user_query: str) -> str:
        """Generate answer for user query using RAG."""
        # Retrieve relevant documents
        relevant_docs = self.retrieval_client.retrieve(user_query)

        # Construct prompt with retrieved context
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"

        # Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # Initialize pipeline
    mongo_uri = "mongodb://localhost:27017"
    qdrant_host = "localhost"
    qdrant_port = 6333
    collection_name = "documents"

    # Initialize main pipeline
    rag_pipeline = RagPipeline(
        mongo_uri=mongo_uri,
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )

    # Process batch documents
    rag_pipeline.process_batch()

    # Initialize retrieval client
    retrieval_client = RetrievalClient(
        qdrant_client=rag_pipeline.qdrant_client,
        collection_name=collection_name,
        embeddings=rag_pipeline.embeddings,
    )

    # Initialize inference pipeline
    inference_pipeline = InferencePipeline("your_model_path", retrieval_client)

    # Example usage
    user_query = "What is the capital of France?"
    answer = inference_pipeline.generate_answer(user_query)
    print(f"Question: {user_query}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
