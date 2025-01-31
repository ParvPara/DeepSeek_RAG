import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from llama_parse import LlamaParse
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle nested async loops
nest_asyncio.apply()

load_dotenv()

# Get Qdrant cloud credentials from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "DEEPSEEK_COLLECTION"

# Initialize global variables
embeddings = OpenAIEmbeddings()
client = None

parsing_instruction = """You are parsing a business document. Maintain the original document structure and metadata. Make sure to take note of graphs, tables and sales data."""

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    parsing_instruction=parsing_instruction,
    result_type="markdown"
)

async def aload_documents():
    """Asynchronously load and parse documents from the data directory."""
    docs_dir = "./data"
    os.makedirs(docs_dir, exist_ok=True)

    files = [
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if f.endswith(('.pdf', '.docx', '.txt'))  
    ]
    
    docs = []
    for file in files:
        try:
            # Run the parsing in an executor to avoid blocking
            parsed_doc = await asyncio.get_event_loop().run_in_executor(
                None, parser.load_data, str(file)
            )
            converted_docs = [
                Document(
                    page_content=doc.text if hasattr(doc, 'text') else str(doc),
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                )
                for doc in parsed_doc
            ]
            docs.extend(converted_docs)
            print(f"Successfully processed {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    return docs

def load_documents():
    """Synchronous wrapper for document loading."""
    return asyncio.get_event_loop().run_until_complete(aload_documents())

async def aprocess_documents(docs):
    """Asynchronously process documents and store them in the vector database."""
    global client
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=100,
        encoding_name="cl100k_base"
    )
    doc_splits = text_splitter.split_documents(docs)
    
    # Create Qdrant client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Get embeddings for documents
    texts = [doc.page_content for doc in doc_splits]
    metadatas = [doc.metadata for doc in doc_splits]
    
    # Run embeddings in executor to avoid blocking
    embeddings_vectors = await asyncio.get_event_loop().run_in_executor(
        None, embeddings.embed_documents, texts
    )
    
    # Create or recreate collection
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(embeddings_vectors[0]),
            distance=models.Distance.COSINE
        )
    )
    
    # Upload documents with their embeddings
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={"text": text, "metadata": metadata}
            )
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings_vectors, metadatas))
        ]
    )
    
    return client

def process_documents(docs):
    """Synchronous wrapper for document processing."""
    return asyncio.get_event_loop().run_until_complete(aprocess_documents(docs))

async def aretrieve_similar(query: str, k: int = 4):
    """Asynchronously retrieve similar documents from the vector store."""
    global client
    
    # Initialize client if not already done
    if client is None:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Get query embedding
    query_vector = await asyncio.get_event_loop().run_in_executor(
        None, embeddings.embed_query, query
    )
    
    # Search in Qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k
    )
    
    # Return documents
    return [
        Document(
            page_content=hit.payload["text"],
            metadata=hit.payload["metadata"]
        )
        for hit in results
    ]

def retrieve_similar(query: str, k: int = 4):
    """Synchronous wrapper for document retrieval."""
    return asyncio.get_event_loop().run_until_complete(aretrieve_similar(query, k))

# Initialize documents on module load
docs = load_documents()
if docs:
    client = process_documents(docs)
else:
    print("No documents found in data directory. Vector store will be created when documents are added.")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Export functions
__all__ = ['retrieve_similar', 'process_documents', 'load_documents']