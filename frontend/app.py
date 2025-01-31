import streamlit as st
import requests
import json
from typing import List, Dict
import os
import shutil
import time

# Set page config and custom CSS
st.set_page_config(page_title="Document Q&A System", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .model-label {
        color: #0f62fe;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 8px;
    }
    .reasoning-box {
        border: 2px solid #0f62fe;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f0f7ff;
    }
    .response-box {
        border: 2px solid #24a148;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f0fff4;
    }
    .document-box {
        border: 1px solid #8c8c8c;
        border-radius: 4px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f8f9fa;
    }
    .stButton button {
        background-color: #0f62fe;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #0353e9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history and models if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "models" not in st.session_state:
    st.session_state.models = {"deepseek": [], "all": []}

def get_available_models() -> Dict:
    """Get list of available models from backend."""
    try:
        response = requests.get("http://localhost:8000/models")
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("models", {"deepseek": [], "all": []})
        return {"deepseek": [], "all": []}
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {"deepseek": [], "all": []}

def send_query(text: str, reasoning_model: str, k: int = 4) -> Dict:
    """Send query to backend API and return response."""
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={
                "text": text,
                "reasoning_model": reasoning_model,
                "k": k
            }
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_document_list() -> List[str]:
    """Get list of documents from backend."""
    try:
        response = requests.get("http://localhost:8000/documents")
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("documents", [])
        return []
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []

def save_uploaded_files(uploaded_files):
    """Save uploaded files to the data directory."""
    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
        os.makedirs(data_dir, exist_ok=True)
        
        # Clear existing files in data directory
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        saved_files = []
        total_size = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                file_content = uploaded_file.getbuffer()
                total_size += len(file_content)
                f.write(file_content)
            saved_files.append(file_path)
            print(f"Saved file: {file_path} (size: {len(file_content) / 1024:.2f} KB)")
        
        print(f"Total size of uploaded files: {total_size / 1024 / 1024:.2f} MB")
        
        # Add a small delay to ensure files are fully written
        time.sleep(1)
        return saved_files, total_size
    except Exception as e:
        st.error(f"Error saving files: {str(e)}")
        return None, 0

def trigger_ingestion():
    """Trigger the ingestion process in the backend."""
    try:
        with st.status("Processing documents...", expanded=True) as status:
            st.write("Processing your documents...")
            st.write("This may take several minutes. Please wait.")
            
            response = requests.post("http://localhost:8000/ingest")
            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    st.success(f"Successfully processed {data['document_count']} documents")
                    st.rerun()  # Refresh the page to update document list
                else:
                    st.error(f"Error: {data['error']}")
            else:
                st.error("Failed to process documents")
            
    except Exception as e:
        st.error(f"Error communicating with backend: {str(e)}")
        return None

def process_documents():
    """Trigger document processing."""
    try:
        with st.spinner('Processing documents... This may take a while for large documents.'):
            response = requests.post("http://localhost:8000/ingest")
            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    st.success(f"Successfully processed {data['document_count']} documents")
                    st.rerun()  # Refresh the page to update document list
                else:
                    st.error(f"Error: {data['error']}")
            else:
                st.error("Failed to process documents")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# App title with styling
st.title("DeepSeek RAG")
st.markdown("---")

# Create two columns for the document section
col1, col2 = st.columns([3, 1])

# Document section
with col1:
    st.markdown("### Current Documents")
    documents = get_document_list()
    if documents:
        for doc in documents:
            st.markdown(f'<div class="document-box">📄 {doc}</div>', unsafe_allow_html=True)
    else:
        st.info("No documents available")

# Refresh button in the second column
with col2:
    st.markdown("### Actions")
    if st.button("🔄", key="refresh_docs"):
        st.rerun()

# Process documents button
if st.button("📥 Process Documents", key="process_docs"):
    process_documents()

st.markdown("---")

# Model selection and refresh
col3, col4 = st.columns([3, 1])

with col3:
    # Update models in session state
    st.session_state.models = get_available_models()
    
    # Model selection
    reasoning_model = st.selectbox(
        "Select Reasoning Model",
        options=st.session_state.models["deepseek"],
        key="reasoning_model"
    )

with col4:
    st.markdown("###")  # Add some spacing
    if st.button("🔄", key="refresh_models"):
        st.session_state.models = get_available_models()
        st.rerun()

# Query input
st.markdown("### Ask a Question")
query = st.text_area("Enter your question:", height=100)

# Submit button
if st.button("Submit Question", key="submit_question"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner('Processing... This may take a while for complex queries or large documents.'):
            result = send_query(query, reasoning_model)
            if result.get("success", False):
                # Display reasoning from DeepSeek
                st.markdown(f'<div class="model-label">🤔 Reasoning Process (using {reasoning_model})</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="reasoning-box">{result["reasoning"]}</div>', unsafe_allow_html=True)
                
                # Display response from GPT
                st.markdown('<div class="model-label">✨ Final Answer (using GPT-4o-mini)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="response-box">{result["response"]}</div>', unsafe_allow_html=True)
                
            else:
                st.error(f"Error: {result.get('error', 'Unknown error occurred')}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 