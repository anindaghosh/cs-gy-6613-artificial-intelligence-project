import gradio as gr
from transformers import pipeline
import ollama

# Initialize Hugging Face model pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define pre-populated questions
questions = [
    "What is ROS2 and its main features?",
    "How does NAV2 work in robot navigation?",
    "Explain MoveIt2's motion planning capabilities.",
    "What are the benefits of using Gazebo for simulations?",
]


# Interaction function with the RAG system
def answer_question(selected_question):
    # Example: Using Ollama (modify as needed)
    # ollama_response = ollama.chat(prompt=selected_question, model="llama2")
    ollama_response = ollama.chat(selected_question)
    hf_response = qa_pipeline(
        question=selected_question, context="Provide a detailed context if available."
    )

    return {
        "Ollama Response": ollama_response.get("text", "N/A"),
        "Hugging Face Response": hf_response.get("answer", "N/A"),
    }


# Define Gradio app layout
def build_app():
    with gr.Blocks() as app:
        gr.Markdown("# RAG System Interactive Demo")

        # Dropdown for questions
        question_dropdown = gr.Dropdown(
            choices=questions, label="Select a question to ask the RAG system"
        )

        # Output area
        output = gr.JSON(label="System Responses")

        # Submit button
        gr.Button("Get Answer").click(
            answer_question, inputs=question_dropdown, outputs=output
        )

    return app


# Launch the app
if __name__ == "__main__":
    gr_app = build_app()
    gr_app.launch(server_port=8000)
