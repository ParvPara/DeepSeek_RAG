from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from backend.chain_rag import process_query, list_models
from fastapi.middleware.cors import CORSMiddleware
import backend.ingestion as ingestion
from backend.file_watcher import setup_file_watcher
import traceback
import os

app = FastAPI()

# Set up file watcher
document_handler = setup_file_watcher()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure to allow unlimited request body size
@app.middleware("http")
async def add_no_limits(request: Request, call_next):
    """Remove size limits on requests."""
    response = await call_next(request)
    return response

class Query(BaseModel):
    text: str
    k: int = 4
    reasoning_model: str = "deepseek"

@app.post("/query")
async def process_user_query(query: Query):
    result = process_query(
        query.text,
        reasoning_model=query.reasoning_model,
        k=query.k
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.get("/models")
async def get_available_models():
    """Get list of available models."""
    try:
        models = list_models()
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/documents")
async def list_documents():
    """List all documents in the data directory."""
    try:
        files = document_handler.update_files()
        return {
            "success": True,
            "documents": [os.path.basename(f) for f in files]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/ingest")
async def ingest_documents():
    """Endpoint to trigger document ingestion process."""
    try:
        # Get absolute path to data directory
        data_dir = os.path.abspath("./data")
        print(f"Looking for documents in: {data_dir}")
        
        # Update file list
        files = document_handler.update_files()
        
        if not files:
            return {
                "success": False,
                "error": f"No valid documents found in {data_dir}. Please upload PDF, DOCX, or TXT files."
            }
        
        print(f"Found {len(files)} files to process: {files}")
        
        # Load and process the documents
        try:
            # First load the documents
            docs = ingestion.load_documents()
            if not docs:
                return {
                    "success": False,
                    "error": "No documents could be loaded from the files"
                }
            
            # Then process them
            client = ingestion.process_documents(docs)
            return {
                "success": True,
                "message": f"Successfully processed {len(files)} documents",
                "document_count": len(files),
                "processed_files": [os.path.basename(f) for f in files]
            }
        except Exception as e:
            print(f"Error during document processing: {str(e)}")
            return {
                "success": False,
                "error": "Failed to process documents",
                "details": str(e)
            }
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error during ingestion: {error_details}")
        return {
            "success": False,
            "error": str(e),
            "details": error_details
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 