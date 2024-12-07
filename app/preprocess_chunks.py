from preprocessing.chunk import split_into_chunks, get_context_from_mongo
from helpers.mongo_client import get_mongo_client


def preprocess_chunks():

    # Connect to MongoDB
    mongo_client = get_mongo_client()
    db = mongo_client["rag"]
    collection = db["rag_cleaned_data"]

    # get context from my own data in mongo
    posts = get_context_from_mongo(collection=collection, platform="github")

    # final_content = ""

    final_chunked_content = []

    for post in posts:
        content = post["cleaned_content"]
        context_chunks = split_into_chunks(content)
        # final_content += content

        for chunk in context_chunks:
            final_chunked_content.append(
                {"content": chunk, "platform": post["platform"], "url": post["url"]}
            )

    # print(context)

    return final_chunked_content
