import os
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

# --- Configuration ---
# The folder where your course documents are stored.
DATA_PATH = "data"
# The folder where the vector database will be stored.
CHROMA_PATH = "chroma_db"
# The embedding model to use. Make sure you have pulled this with "ollama pull mxbai-embed-large"
EMBEDDING_MODEL = "nomic-embed-text"

# A list of relevant websites to scrape for additional information.
URLS_TO_SCRAPE = [
    "https://owasp.org/www-project-top-ten/",
    "https://www.cisa.gov/shields-up",
    "https://attack.mitre.org/techniques/enterprise/"
    # Add more URLs for your Cybersecurity T-Level here.
    # For example, websites from NIST, NCSC (UK), or specific tool documentation.
]

def main():
    """
    Main function to orchestrate the document ingestion process.
    """
    print("--- Starting Document Ingestion Process ---")
    
    # 1. Load documents from all sources.
    documents = load_documents()
    if not documents:
        print("No new documents to process. Exiting.")
        return

    # 2. Split the loaded documents into smaller chunks.
    chunks = split_documents(documents)
    
    # 3. Save the document chunks to the vector store.
    save_to_vector_store(chunks)
    
    print("\n--- Ingestion Complete! ---")
    print(f"Your knowledge base is now ready in the '{CHROMA_PATH}' directory.")
    print("You can now run the main application ('app.py').")

def load_documents():
    """
    Loads documents from the local 'data' directory and scrapes specified URLs.
    Supports PDF, TXT, MD, DOCX, PPTX, and CSV files.
    """
    documents = []
    print(f"\n1. Loading documents from '{DATA_PATH}' and scraping URLs...")
    
    # Load local files
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"  ✔ Loaded PDF: {filename}")
            elif filename.endswith(('.txt', '.md')):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                print(f"  ✔ Loaded Text: {filename}")
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
                print(f"  ✔ Loaded DOCX: {filename}")
            elif filename.endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(file_path)
                documents.extend(loader.load())
                print(f"  ✔ Loaded PPTX: {filename}")
            elif filename.endswith('.csv'):
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
                print(f"  ✔ Loaded CSV: {filename}")
        except Exception as e:
            print(f"  ✖ Failed to load {filename}: {e}")
            
    # Scrape and load web pages
    for url in URLS_TO_SCRAPE:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
            print(f"  ✔ Scraped and loaded: {url}")
        except Exception as e:
            print(f"  ✖ Failed to scrape {url}: {e}")
            
    return documents

def split_documents(documents):
    """
    Splits the documents into smaller chunks for efficient processing.
    """
    print("\n2. Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # The size of each chunk in characters.
        chunk_overlap=200, # The number of characters to overlap between chunks.
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_vector_store(chunks):
    """
    Initializes the embedding model and saves the document chunks to ChromaDB.
    """
    print("\n3. Creating embeddings and saving to vector store...")
    
    # Initialize the Ollama embedding model.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Create a new ChromaDB store from the document chunks.
    # This will create and persist the database in the CHROMA_PATH folder.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"   Saved {len(chunks)} chunks to '{CHROMA_PATH}'.")

if __name__ == "__main__":
    main()