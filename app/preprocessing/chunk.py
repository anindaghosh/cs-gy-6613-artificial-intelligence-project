import re
import re

from loguru import logger

from clearml import Task

from app.helpers.mongo_client import get_mongo_client


def get_context_from_mongo(collection, platform):

    posts = collection.find({"platform": platform})

    # final_content = ""

    return posts


def split_into_chunks(context, max_tokens=2000):
    words = context.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def etl_chunk():

    task = Task.init(project_name="cs-gy-6613-rag", task_name="etl_chunker")

    logger.info("Starting chunking process of cleaned docs...")

    # Connect to MongoDB
    mongo_client = get_mongo_client()
    db = mongo_client["rag"]
    collection = db["rag_cleaned_data"]
    new_collection = db["rag_chunked_data"]

    posts = collection.find({})

    # final_chunked_content = []

    for post in posts:
        content = post["cleaned_content"]
        context_chunks = split_into_chunks(content)
        # final_content += content

        for chunk in context_chunks:
            new_collection.insert_one(
                {"content": chunk, "platform": post["platform"], "url": post["url"]}
            )

        logger.info(f"Chunked content for post: {post['url']}")

    logger.info("Finished chunking the documents.")
    task.close()


if __name__ == "__main__":

    etl_chunk()
