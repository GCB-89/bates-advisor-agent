"""
PDF Ingestion Script - Loads Bates Technical College Catalog into Vector Database

This script:
1. Reads the PDF file
2. Splits it into manageable chunks
3. Creates embeddings (vector representations)
4. Stores in ChromaDB for fast similarity search

Run this ONCE after setting up your project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()


def ingest_pdf():
    """
    Main ingestion function that processes the PDF and creates the vector database.
    """
    print("🚀 Starting PDF ingestion process...")
    print("=" * 60)
    
    # Define paths
    pdf_path = Path("data/BatesTech2025-26Catalog.pdf")
    chroma_db_path = "./chroma_db"
    
    # Check if PDF exists
    if not pdf_path.exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        print("Please place BatesTech2025-26Catalog.pdf in the data/ folder")
        return
    
    # Step 1: Load the PDF
    print(f"\n📄 Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    
    # Step 2: Split documents into chunks
    print("\n✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Each chunk is ~1000 characters
        chunk_overlap=200,      # 200 character overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Split on paragraphs first, then sentences
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks")
    
    # Show sample chunk for learning
    print(f"\n📝 Sample chunk (for your reference):")
    print("-" * 60)
    print(f"Content: {chunks[0].page_content[:200]}...")
    print(f"Metadata: {chunks[0].metadata}")
    print("-" * 60)
    
    # Step 3: Create embeddings and store in ChromaDB
    print("\n🧠 Creating embeddings and storing in vector database...")
    print("(This may take 2-5 minutes for 459 pages...)")
    # Initialize Google embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
    )
    
    
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_db_path,
        collection_name="bates_catalog"
    )
    
    print(f"✅ Vector database created at: {chroma_db_path}")
    
    # Step 4: Test the vector store
    print("\n🔍 Testing vector store with a sample query...")
    test_query = "What programs does Bates Tech offer?"
    results = vectorstore.similarity_search(test_query, k=2)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Found {len(results)} relevant chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n  Result {i} (Page {doc.metadata.get('page', 'N/A')}):")
        print(f"  {doc.page_content[:150]}...")
    
    print("\n" + "=" * 60)
    print("🎉 Success! PDF ingestion complete!")
    print("\nNext step: Run 'python src/chat.py' to start chatting with your advisor!")
    print("=" * 60)


if __name__ == "__main__":
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not found in environment")
        print("Please create a .env file with your Google API key")
        print("Get your key at: https://aistudio.google.com/apikey")
        exit(1)
    
    ingest_pdf()
