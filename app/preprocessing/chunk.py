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
