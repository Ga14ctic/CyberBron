import os
import yaml
import logging
from tqdm import tqdm
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
def load_config():
    """Load configuration from config.yaml with fallback to defaults."""
    config_path = "config.yaml"
    default_config = {
        "models": {"embeddings": "nomic-embed-text"},
        "paths": {
            "data": "data",
            "chroma_db": "chroma_db"
        },
        "rag": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        "web_sources": [
            "https://owasp.org/www-project-top-ten/",
            "https://www.cisa.gov/shields-up",
            "https://attack.mitre.org/techniques/enterprise/"
        ]
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded from config.yaml")
                return config
        else:
            logger.warning("config.yaml not found, using default configuration")
            return default_config
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        return default_config

CONFIG = load_config()
DATA_PATH = CONFIG["paths"]["data"]
CHROMA_PATH = CONFIG["paths"]["chroma_db"]
EMBEDDING_MODEL = CONFIG["models"]["embeddings"]
URLS_TO_SCRAPE = CONFIG.get("web_sources", [])

def main():
    """
    Main function to orchestrate the document ingestion process.
    """
    logger.info("Starting Document Ingestion Process")
    print("\n" + "="*60)
    print("  CyberBron Document Ingestion")
    print("="*60 + "\n")
    
    # Ensure data directory exists
    if not os.path.exists(DATA_PATH):
        logger.info(f"Creating data directory: {DATA_PATH}")
        os.makedirs(DATA_PATH)
        print(f"✓ Created data directory: {DATA_PATH}")
        print(f"  Please add your documents to '{DATA_PATH}' and run this script again.")
        return
    
    # 1. Load documents from all sources.
    documents = load_documents()
    if not documents:
        print("\n⚠️  No documents found to process.")
        print(f"   Please add documents to the '{DATA_PATH}' directory.")
        return

    # 2. Split the loaded documents into smaller chunks.
    chunks = split_documents(documents)
    
    # 3. Save the document chunks to the vector store.
    save_to_vector_store(chunks)
    
    print("\n" + "="*60)
    print("  ✓ Ingestion Complete!")
    print("="*60)
    print(f"  Your knowledge base is ready in '{CHROMA_PATH}'")
    print("  You can now run: streamlit run app.py")
    print("="*60 + "\n")
    logger.info("Ingestion process completed successfully")

def load_documents():
    """
    Loads documents from the local 'data' directory and scrapes specified URLs.
    Supports PDF, TXT, MD, DOCX, PPTX, and CSV files.
    """
    documents = []
    print(f"\n📂 Step 1: Loading documents from '{DATA_PATH}'...")
    
    # Get list of files
    try:
        files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    except Exception as e:
        logger.error(f"Error listing files in {DATA_PATH}: {e}")
        print(f"  ✖ Error accessing {DATA_PATH}: {e}")
        return documents
    
    if not files:
        logger.warning(f"No files found in {DATA_PATH}")
        print(f"  ⚠️  No files found in '{DATA_PATH}'")
        return documents
    
    # Load local files with progress bar
    print(f"  Found {len(files)} file(s)")
    successful_loads = 0
    
    for filename in tqdm(files, desc="  Loading files", unit="file"):
        file_path = os.path.join(DATA_PATH, filename)
        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                successful_loads += 1
                logger.info(f"Loaded PDF: {filename}")
            elif filename.endswith(('.txt', '.md')):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                successful_loads += 1
                logger.info(f"Loaded Text: {filename}")
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
                successful_loads += 1
                logger.info(f"Loaded DOCX: {filename}")
            elif filename.endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(file_path)
                documents.extend(loader.load())
                successful_loads += 1
                logger.info(f"Loaded PPTX: {filename}")
            elif filename.endswith('.csv'):
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
                successful_loads += 1
                logger.info(f"Loaded CSV: {filename}")
            else:
                logger.debug(f"Skipped unsupported file: {filename}")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            print(f"  ✖ Failed: {filename} - {str(e)[:50]}")
    
    print(f"  ✓ Successfully loaded {successful_loads}/{len(files)} file(s)")
    
    # Scrape and load web pages
    if URLS_TO_SCRAPE:
        print(f"\n🌐 Loading {len(URLS_TO_SCRAPE)} web source(s)...")
        for url in tqdm(URLS_TO_SCRAPE, desc="  Scraping URLs", unit="url"):
            try:
                loader = WebBaseLoader(url)
                web_docs = loader.load()
                documents.extend(web_docs)
                logger.info(f"Scraped: {url}")
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                print(f"  ✖ Failed: {url[:50]}... - {str(e)[:50]}")
    
    print(f"  ✓ Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """
    Splits the documents into smaller chunks for efficient processing.
    """
    print(f"\n✂️  Step 2: Splitting documents into chunks...")
    
    chunk_size = CONFIG["rag"]["chunk_size"]
    chunk_overlap = CONFIG["rag"]["chunk_overlap"]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"  ✓ Split {len(documents)} document(s) into {len(chunks)} chunk(s)")
    print(f"    (chunk_size={chunk_size}, overlap={chunk_overlap})")
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    return chunks

def save_to_vector_store(chunks):
    """
    Initializes the embedding model and saves the document chunks to ChromaDB.
    Uses batch processing for better memory efficiency.
    """
    print(f"\n💾 Step 3: Creating embeddings and saving to vector store...")
    print(f"  Using embedding model: {EMBEDDING_MODEL}")
    
    try:
        # Initialize the Ollama embedding model
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Process in batches to prevent memory issues with large document sets
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        if len(chunks) > batch_size:
            print(f"  Processing {len(chunks)} chunks in {total_batches} batch(es) of {batch_size}")
            
            # Create initial batch
            vector_store = Chroma.from_documents(
                documents=chunks[:batch_size],
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
            
            # Add remaining batches
            for i in tqdm(range(batch_size, len(chunks), batch_size), 
                         desc="  Processing batches", 
                         unit="batch"):
                batch = chunks[i:i + batch_size]
                vector_store.add_documents(batch)
        else:
            # For smaller sets, process all at once
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
        
        print(f"  ✓ Saved {len(chunks)} chunk(s) to '{CHROMA_PATH}'")
        logger.info(f"Saved {len(chunks)} chunks to vector store")
        
    except Exception as e:
        logger.error(f"Error saving to vector store: {e}")
        print(f"  ✖ Error: {e}")
        raise

if __name__ == "__main__":
    main()