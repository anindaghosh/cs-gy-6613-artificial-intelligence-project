from openai import OpenAI
import json

from helpers.mongo_client import get_mongo_client
from configs import OPENAI_API_KEY

# Set up OpenAI API key

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_context_from_mongo(collection):

    posts = collection.find({"platform": "medium"})

    final_content = ""

    for post in posts:
        content = post["cleaned_content"]
        final_content += content

    return final_content


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


# Function to generate questions
def generate_questions(context, num_questions=5):
    # prompt = (
    #     f"Given the following content, generate {num_questions} questions that are relevant and helpful. Ensure to not add any numbering or bullets before the question:\n\n"
    #     f"Context: {context}\n\n"
    #     f"Questions:"
    # )

    prompt = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
        must ask to write about a specific topic contained in the context. Each answer \
        must provide a relevant paragraph based on the information found in the \
        context. Only use concepts from the context to generate the instructions. \
        Instructions must never explicitly mention a context, a system, a course, or an extract. \
        Instructions must be self-contained and general. \
        Answers must imitate the writing style of the context. \
            
        Example instruction: Explain the concept of an LLM Twin. \
        Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
        It's designed to write just like you by incorporating these elements into a language model. \
        The idea is to create a digital replica of your writing habits using advanced AI techniques. \

        Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
        Do not add any extra characters and provide your response in JSON format with the following structure:
        [
            {"instruction": "...", "answer": "..."},
            ...
        ]

        Extract:
        """ + """{context}""".format(
        context=context
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1200,
    )

    # Extract the questions from the response
    # questions = response["choices"][0]["message"]["content"].strip().split("\n")

    # print(response)

    # print(response.choices[0].message.content.strip())

    result = response.choices[0].message.content.strip()

    json_body = result.replace("```json", "").replace("```", "")

    questions = json.loads(json_body)

    # print(questions)

    # questions = [q.strip() for q in response.split("\n\n") if q.strip()]
    # return questions

    return questions


if __name__ == "__main__":

    # Connect to MongoDB
    mongo_client = get_mongo_client()
    db = mongo_client["rag"]
    collection = db["rag_cleaned_data"]

    instruct_set = db["instruct_set"]

    # Example usage
    sample_context = (
        "The linux-real-time-kernel-builder is a tool to configure and build real-time kernels optimized for ROS2. "
        "It provides automation for downloading and setting up the kernel with required configurations for real-time applications."
    )

    # get context from my own data in mongo
    context = get_context_from_mongo(collection)

    # print(context)

    context_chunks = split_into_chunks(context)

    # print(context_chunks)

    all_questions = []

    print(f"Length of chunks: {len(context_chunks)}")

    for chunk in context_chunks:
        questions = generate_questions(chunk)
        # # print(questions)
        # all_questions.extend(questions)

        # print(questions)

        for question in questions:
            print(question)

            # if question is already in the collection, skip
            if instruct_set.find_one({"instruction": question["instruction"]}):
                print(f"Instruction already exists: {question['instruction']}")
                continue

            instruct_set.insert_one(question)

        # break

    # print(all_questions)
