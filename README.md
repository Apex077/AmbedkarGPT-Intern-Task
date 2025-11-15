# AmbedkarGPT - Intern Task

A simple command-line Q&A system that uses RAG (Retrieval-Augmented Generation) to answer questions based on Dr. B.R. Ambedkar's speech content. I've used a more quantized model: `mistral:7b-instruct-q4_0`, since I don't have a dGPU.

## Features

- **Caching**: In case the content of `speech.txt` doesn't change, I've ensured that the same vector store is used, thus improving speed and reducing resource usage.
- **Manual Cache Reset**: In case you require the cache to be reset manually, it can be done by using the `reset` option. It also monitors for changes in speech.txt, so a new vector store can be created if required.
- **Lower Chances of Hallucination**: Chances of hallucination are very low.

## Overview

This system demonstrates the fundamental building blocks of a RAG pipeline:
1. **Text Loading**: Ingests text from `speech.txt`
2. **Text Splitting**: Breaks content into manageable chunks
3. **Embeddings**: Creates vector representations using HuggingFace sentence-transformers
4. **Vector Storage**: Stores embeddings in ChromaDB for efficient retrieval
5. **Question Answering**: Uses Ollama with the Mistral model to generate contextual answers, while reducing hallucinations

## Technical Stack

- **Python**: 3.8+ (Tested on Python 3.13.7, on Arch Linux)
- **Framework**: LangChain for RAG orchestration
- **Vector Database**: ChromaDB (local, no setup required)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Ollama with Mistral 7B (100% free, local)

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running

## Setup Instructions

### Step 1: Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral 7B model (or the model mentioned in the intro, in case you're running this on CPU)
ollama pull mistral

# Test Ollama (optional)
ollama run mistral "Hello, how are you?"
```

### Step 2: Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/Apex077/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run the System

```bash
python main.py
```

## Usage

1. **Start the system**: Run `python main.py`
2. **Wait for initialization**: The system will:
   - Load and split the speech text
   - Download embeddings model (first run only)
   - Create vector store (and look for cached versions on subsequent runs)
   - Setup QA chain
3. **Ask questions**: Type your questions about the speech content
4. **Exit**: Type `quit`, `exit`, or `q` to stop, or `reset` to reset the vector store.

## Example Questions (For this assignment)

- "What is the real remedy according to the speech?"
- "What does the author say about caste?"
- "How does the author compare social reform to gardening?"
- "What is the real enemy mentioned in the speech?"

## How It Works

1. **Text Processing**: The speech is loaded and split into overlapping chunks for better context retrieval
2. **Embedding Creation**: Each chunk is converted to a vector using sentence-transformers
3. **Vector Storage**: Embeddings are stored in ChromaDB for fast similarity search
4. **Question Processing**: When you ask a question:
   - Your question is embedded using the same model
   - Similar text chunks are retrieved from the vector store
   - Retrieved context + your question are sent to Mistral 7B
   - The LLM generates an answer based on the provided context

## Dependencies

- `langchain==1.0.5`: RAG pipeline orchestration
- `langchain-community==0.4.1`: Community integrations for LangChain
- `langchain-ollama==1.0.0`: Ollama integration for LangChain
- `langchain-huggingface==1.0.1`: HuggingFace integration for LangChain
- `langchain-text-splitters==1.0.0`: Text splitting utilities
- `langchain-core==1.0.5`: Core LangChain functionality
- `chromadb==1.3.4`: Local vector database
- `sentence-transformers==5.1.2`: Text embeddings

## Troubleshooting

### Common Issues

1. **Ollama not found**: Ensure Ollama is installed and in your PATH
2. **Mistral model missing**: Run `ollama pull mistral`
3. **Permission errors**: Make sure you have write permissions in the project directory
4. **Memory issues**: Ensure you have at least 4GB RAM available for Mistral 7B

### Error Messages

- **"Ollama not running"**: Start Ollama service or run `ollama serve`
- **"Model not found"**: Pull the Mistral model with `ollama pull mistral`
- **"Import errors"**: Activate virtual environment and reinstall requirements

## Performance Notes

- **First run**: May take longer due to model downloads
- **Subsequent runs**: Faster as models and vector store are cached
- **Response time**: Depends on your hardware (CPU/GPU availability)

## No API Keys Required

- Ollama (local LLM)
- HuggingFace embeddings/sentence-transformers (local)
- ChromaDB (local vector store)
- No external API calls
- No accounts needed
- Completely free

## Limitations

- Answers are limited to the content in `speech.txt`
- Response quality depends on question clarity and context relevance
- Local processing means performance varies by hardware