# Document RAG System

A Retrieval Augmented Generation (RAG) system that allows users to chat with large documents.

## Features

- Upload and process large text documents
- Customizable chunking strategy
- Support for both FAISS (in-memory) and ChromaDB (persistent) vector stores
- Configurable LLM parameters (model, temperature)
- Streamlit-based chat interface
- Dockerized for easy deployment

## Architecture

![RAG Architecture Diagram](https://miro.medium.com/v2/resize:fit:1400/1*5v5wQRFZUU3J0Q4Qh7KvVg.png)

1. **Document Processing**: The document is loaded and split into chunks using LangChain's text splitter.
2. **Embedding Generation**: Each chunk is converted to a vector embedding using either OpenAI or HuggingFace embeddings.
3. **Vector Storage**: Embeddings are stored in a vector database (FAISS or ChromaDB).
4. **Query Processing**: User questions are converted to embeddings and used to retrieve relevant document chunks.
5. **Response Generation**: Retrieved chunks and the question are sent to an LLM to generate a contextual answer.

## Setup

### Prerequisites

- Docker
- Docker Compose
- OpenAI API key (if using OpenAI embeddings)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KALEMA-ROGERS/Document-RAG-System
   cd document-rag-system