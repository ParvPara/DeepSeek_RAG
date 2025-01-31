import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import asyncio
import nest_asyncio
from backend import ingestion

# Apply nest_asyncio to handle nested async loops
nest_asyncio.apply()

class DocumentHandler(FileSystemEventHandler):
    def __init__(self):
        self.files = set()
        self.update_files()
        self.processing = False
    
    def update_files(self):
        """Update the list of files in the data directory."""
        data_dir = os.path.abspath("./data")
        if os.path.exists(data_dir):
            self.files = {
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(('.pdf', '.docx', '.txt'))
            }
        return list(self.files)
    
    async def process_documents(self):
        """Process documents asynchronously."""
        if self.processing:
            return
        
        try:
            self.processing = True
            print("Processing documents...")
            docs = ingestion.load_documents()
            if docs:
                client = ingestion.process_documents(docs)
                print("Documents processed successfully")
            else:
                print("No documents to process")
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
        finally:
            self.processing = False
    
    def handle_file_change(self):
        """Handle file changes by triggering document processing."""
        asyncio.create_task(self.process_documents())
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.pdf', '.docx', '.txt')):
            self.files.add(event.src_path)
            self.handle_file_change()
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path in self.files:
            self.files.remove(event.src_path)
            self.handle_file_change()
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.pdf', '.docx', '.txt')):
            if event.src_path not in self.files:
                self.files.add(event.src_path)
            self.handle_file_change()
    
    def on_moved(self, event):
        if not event.is_directory:
            if event.src_path in self.files:
                self.files.remove(event.src_path)
            if event.dest_path.endswith(('.pdf', '.docx', '.txt')):
                self.files.add(event.dest_path)
            self.handle_file_change()

def setup_file_watcher():
    """Set up the file watcher for the data directory."""
    data_dir = os.path.abspath("./data")
    os.makedirs(data_dir, exist_ok=True)
    
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, data_dir, recursive=False)
    observer.start()
    
    return event_handler 