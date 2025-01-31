# DeepSeek RAG Project

This project combines RAG (Retrieval Augmented Generation) with DeepSeek and GPT models to provide intelligent responses based on your document collection.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
LLAMA_CLOUD_API_KEY=your_llama_parse_api_key
```

3. Install and start Ollama with DeepSeek:
```bash
# Install Ollama from https://ollama.ai/
# Pull DeepSeek model
ollama pull deepseek-coder:1.3b
```

4. Place your documents in the `data` directory (supports PDF, DOCX, and TXT files)

## Usage

1. Start the ingestion process to load documents into the vector store:
```bash
python backend/ingestion.py
```

2. Start the FastAPI backend server:
```bash
uvicorn backend.app:app --reload
```

3. Start the Streamlit frontend (in a new terminal):
```bash
streamlit run frontend/app.py
```

The Streamlit interface will open in your default web browser at `http://localhost:8501`. You can now:
- Ask questions about your documents
- View chat history
- See the context and reasoning behind each response
- Clear chat history when needed

## API Usage (for developers)

If you prefer to use the API directly:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "your question here", "k": 4}'
```

## How it Works

1. **Document Ingestion**: Documents are processed using LlamaParse and stored in Qdrant vector store
2. **Query Processing**:
   - Retrieves relevant context using RAG
   - DeepSeek model reasons about the query using the context
   - GPT model generates a final response based on DeepSeek's reasoning

## API Endpoints

- `POST /query`: Process a query through the RAG + DeepSeek + GPT chain
  - Parameters:
    - `text`: The query text
    - `k`: Number of relevant documents to retrieve (default: 4)
- `GET /health`: Health check endpoint

## Project Structure

```
.
├── backend/
│   ├── app.py           # FastAPI server
│   ├── chain_rag.py     # RAG + DeepSeek + GPT chain implementation
│   └── ingestion.py     # Document ingestion script
├── frontend/
│   └── app.py           # Streamlit chat interface
├── data/                # Place your documents here
├── requirements.txt     # Project dependencies
└── .env                # Environment variables
``` 