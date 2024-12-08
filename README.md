# Artificial Intelligence Project - Finetuned RAG Systems Engineering (CS-GY 6613)

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Team Members
1. **Aninda Ghosh** (ag10293)
- **GitHub**: anindaghosh
- **HuggingFace**: anindaghosh
2. **Samrudhi Baldota** (sb10212)
- **GitHub**: samrudhibaldota
- **HuggingFace**: anindaghosh

## Introduction
The Retrieval-Augmented Generation (RAG) system we are developing aims to provide a powerful, domain-specific solution for ROS2 robotics developers, particularly those working on the navigation stack of autonomous robots with egomotion. The purpose of this project is to create an intelligent system that can efficiently retrieve and generate answers related to specific questions about ROS2, navigation, motion planning, and simulation in robotics. The system will combine retrieval-based and generation-based models to offer highly relevant, context-aware responses that assist developers in their work.

The goal of the project :
1.Provide a Robust Information Retrieval System : To build a vector search engine using Qdrant to retrieve domain related data from a large corpus of information.

2. Domain-Specific Language Generation: To Fine-tune a pre-trained language model to understand and generate accurate responses based on the retrieved documents, focusing on ROS2 middleware, Nav2 navigation, Movit2 motion planning, and Gazebo simulation.

3.ETL and Featurization Pipelines: To Design a pipeline to ingest various sources of data (e.g., GitHub- ROS2 documentation, YouTube videos , Medium) and convert this raw data into vector embeddings that are suitable for use by the RAG system. 

4.User Interaction via Gradio: To Build an interactive Gradio app where users can query the system, select predefined questions related to the subdomains, and receive domain-specific answers in real-time.

5. Scalable, Modular Architecture: Use Docker to containerize the system, ensuring reproducibility and scalability across different development environments. Integrate ClearML for experiment tracking and orchestration, ensuring that the system's performance is continuously monitored and improved.

## Project Overview
The Retrieval-Augmented Generation (RAG) system is a complex architecture composed of several key components that work together to enable an intelligent, domain-specific question-answering system for ROS2 robotics developers. Each component is designed to serve a distinct purpose while interacting seamlessly with the others. Below is an overview of the main components and their functionality:

1. Environment and Infrastructure (Docker Compose)
Purpose: Set up a consistent, reproducible development environment that includes all necessary services for running the RAG system.

Components:
App: A containerized service responsible for managing and serving machine learning models, interacting with APIs (e.g., Hugging Face Hub), and processing data.
MongoDB: A NoSQL database used to store both raw data (from the ETL pipeline) and featurized data (embeddings) in a flexible, scalable way.
Qdrant: A vector search engine that allows efficient retrieval of high-dimensional embeddings based on similarity. This component is key to the RAG system’s retrieval functionality.
ClearML: An experiment tracking and orchestration system that helps manage the machine learning pipeline, from data ingestion and model training to evaluation and deployment.
Functionality:Use Docker Compose to ensure that all the services (App, MongoDB, Qdrant, ClearML) are containerized and orchestrated together.
Provide an isolated environment where developers can easily replicate the project setup across different machines.

2. ETL Pipeline (Extract, Transform, Load)
Purpose: Extract relevant data from multiple sources (e.g., GitHub repositories, YouTube videos), transform it into a usable format, and load it into the database for later use.

Components:
Data Sources: This includes ROS2 documentation (hosted on GitHub), YouTube video transcripts, and potentially other media related to ROS2 robotics, navigation, motion planning, and simulation.
ClearML Orchestrator: Used to orchestrate the ETL pipeline, allowing for efficient scheduling and tracking of the data ingestion process.
MongoDB Database: Stores the raw, unprocessed data (e.g., video transcripts, documentation) that will later be transformed and featurized.
Functionality:

Extract: Use APIs or scraping techniques to pull ROS2-related documentation and other domain-specific media (e.g., video transcripts) from online platforms.
Transform: Clean and preprocess the raw data (e.g., parsing and structuring text from video transcripts or markdown files).
Load: Insert the processed raw data into MongoDB for easy retrieval during the featurization and retrieval stages.

3. Featurization Pipeline
Purpose: Convert raw textual data into numerical vector embeddings that can be used by the retrieval and generation models in the RAG system.

Components:
Sentence Transformers: A pre-trained model that converts raw text into vector embeddings. These embeddings capture the semantic meaning of the text and allow for similarity-based retrieval.
Qdrant Vector Database: Stores these embeddings and provides fast, similarity-based retrieval to support the generation model in finding relevant data.

Functionality:Featurization: Use a model like SentenceTransformers to encode text from the ROS2 documentation and video transcripts into high-dimensional vectors.
Storing Embeddings: Insert these embeddings into Qdrant, allowing for efficient nearest-neighbor search when a query is made.
Indexing and Searching: The featurized data in Qdrant is indexed, enabling fast retrieval based on the similarity between the query vector and the stored vectors.

4. Retrieval System (Vector Search with Qdrant)
Purpose: Retrieve relevant documents from the database based on the similarity between the query and stored embeddings.

Components:
Qdrant: The vector search engine that indexes embeddings and allows for fast similarity searches.
Functionality: When a user asks a question, the system generates an embedding of the query using the same model used for featurization.
Qdrant performs a nearest-neighbor search to find the most relevant documents (or embeddings) from the database based on the query embedding.
The retrieved documents are passed to the generation model to create a coherent, context-aware response.

5. Generation Model (Pre-trained Language Model)
Purpose: Generate human-like, context-aware responses to user queries by synthesizing information from the retrieved documents.

Components:
Pre-trained Language Model: A transformer-based model (e.g., GPT, T5, or a custom fine-tuned model) that generates text based on input prompts.

Functionality:The generation model receives a combination of the user query and the retrieved documents (context) from Qdrant.
It uses this context to generate a response, combining the information from multiple documents to create a coherent and relevant answer.
The model can be fine-tuned on domain-specific data (ROS2, navigation, etc.) to improve accuracy and relevance of the responses.

6. Fine-Tuning 
Purpose: Fine-tune the language model to improve its performance on domain-specific queries related to ROS2 robotics, Gazebo using Llama 3.1.8 pre-trained model.

Components:
Huggingface Transformers: A framework for fine-tuning pre-trained models on custom datasets.
ROS2-specific Dataset: A curated dataset containing domain-specific content (e.g., ROS2 documentation, navigation code snippets, etc.).
Functionality: Use the fine-tuning process to adapt a general-purpose language model to the specific needs of ROS2, improving the quality of responses in the domain.
Fine-tune the model on the preprocessed ROS2-related data to ensure it understands the terminology, concepts, and intricacies of the domain.

7. User Interface 
Purpose: Provide an interactive, user-friendly interface for querying the RAG system.

Components:
Open Web UI: A Python library for creating simple web-based UIs for machine learning models.

Functionality: Users can input questions related to the four ROS2 subdomains (ROS2 middleware, Nav2, Movit2, and Gazebo simulation).
The system retrieves relevant information and generates answers based on the RAG system.
A dropdown menu pre-populates common questions for users to select from, streamlining the interaction process.
The app displays the generated answers and allows users to further refine or modify their queries.

8. Experiment Tracking and Orchestration (ClearML)
Purpose: Track and manage the machine learning experiments, data pipelines, and model evaluations.

Components:
ClearML: An orchestrator and experiment tracking system that records data preprocessing, model training, fine-tuning, and performance evaluations.
Functionality: Track Experiments: Monitor model training, fine-tuning, and evaluation experiments.
Pipeline Orchestration: Schedule and manage ETL and featurization pipelines, ensuring smooth execution and reproducibility.
Performance Monitoring: Track key metrics and model performance to evaluate improvements during fine-tuning or iterative development.


## Installation
Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/anindaghosh/cs-gy-6613-artificial-intelligence-project

# Navigate to the project directory
cd cs-gy-6613-artificial-intelligence-project

# Set up a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


# Run the application
python app.py
```

## Usage
Explain how to use the project, including any necessary commands or steps.

```bash
# Compose the docker file
docker-compose up -d

# Run RAG Jupiter Notebook
Here we must run the Jupiter Notebook -
1. URLs are collected from MongoDB for scraping.
2. Once scraping is completed we clean the data from the url which is stored in Mongo DB of any non-ascii characters.
3. We then chunk the url into pieces and embed chunks into the Vector DB(Qdrant).

# Run Fine tune Jupiter Notebook
1. This notebook setsup a training a pipeline for fine-tuning using the LLaMA 3.1.8 pre-trained model using Low-Rank Adaptation (LoRA)for optimized memory usage and faster training.
2.Instruction-answer pairs are extracted from Gpt based on the data from Qdrant, a vector search engine, and used to create a custom dataset for training the model which is then uploaded to Hugging face.

# Final Result
1. With the help of Open Web UI the user to interact with the RAG system using Ollama and pull our model from HF hub.
2. ClearML experiment tracking and orchestration, ensuring that all training parameters and results are logged for reproducibility.

```

## Features
List the key features of your project.

- Feature 1 - Data Collection from MongoDB and Qdrant
- Feature 2 - Custom Instruction Dataset
- Feature 3 - Fine-Tuning of LLaMA Model using LoRA
- Feature 4 - Training Pipeline with LoRA & Mixed Precision
- Feature 5 - Hugging Face Integration
- Feature 6 - Model deployment using Open Web UI
- Feature 7 - Rag System

## Technologies Used
Mention the technologies and tools used in the project.

- Python
- TensorFlow
- NumPy
- Other relevant technologies

## Contributing
Guidelines for contributing to the project.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request
