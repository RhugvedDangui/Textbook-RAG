"""
Script to ingest textbook PDFs and create a ChromaDB vector store.

Usage:
    python create_vectordb.py                       # Process all PDFs in ./Textbook
    python create_vectordb.py --source ./Textbook   # Specify custom folder
    python create_vectordb.py --reset               # Delete existing DB and recreate
"""

import os
import shutil
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
TEXTBOOK_DIR = os.path.join(os.path.dirname(__file__), "Textbook")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "textbook_rag"


# ── PDF Parsing with PyPDF ────────────────────────────────────────────────────

def load_pdfs(source_dir: str) -> list[Document]:
    """Parse all PDFs in the source directory using PyPDFLoader and return LangChain Documents.

    PyPDFLoader extracts the text layer directly — no image rendering, no OCR.
    This is fast and memory-safe for digital (non-scanned) PDF textbooks.
    Each PDF page becomes a separate Document with 'source' and 'page' metadata.
    """

    pdf_files = list(Path(source_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠  No PDF files found in '{source_dir}'")
        return []

    print(f"📚 Found {len(pdf_files)} PDF(s) in '{source_dir}'")

    all_documents: list[Document] = []

    for pdf_path in pdf_files:
        print(f"\n📖 Processing: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()  # one Document per page, with page metadata

            if not pages:
                print(f"   ⚠  No text extracted from {pdf_path.name}, skipping.")
                continue

            # Stamp every page with a clean source filename
            for page in pages:
                page.metadata["source"] = pdf_path.name

            all_documents.extend(pages)
            total_chars = sum(len(p.page_content) for p in pages)
            print(f"   ✅ {len(pages)} pages extracted ({total_chars:,} characters)")

        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")

    print(f"\n📄 Total pages loaded: {len(all_documents)}")
    return all_documents


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embedding."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"🔪 Split into {len(chunks):,} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── Vector Store Creation ─────────────────────────────────────────────────────

def create_vector_store(chunks: list[Document], db_dir: str) -> Chroma:
    """Create ChromaDB vector store from document chunks."""

    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"💾 Creating ChromaDB at: {db_dir}")

    # ChromaDB has a batch limit; insert in batches of 5000
    BATCH_SIZE = 5000
    vector_store = None

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        print(f"   Embedding batch {start // BATCH_SIZE + 1} ({len(batch)} chunks)...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=db_dir,
                collection_name=COLLECTION_NAME,
            )
        else:
            vector_store.add_documents(batch)

    print(f"✅ Vector store created with {len(chunks):,} vectors")
    return vector_store


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build a ChromaDB vector store from textbook PDFs.")
    parser.add_argument("--source", type=str, default=TEXTBOOK_DIR, help="Path to folder with PDF textbooks")
    parser.add_argument("--reset", action="store_true", help="Delete existing vector store before creating")
    args = parser.parse_args()

    # Optionally reset
    if args.reset and os.path.exists(CHROMA_DB_DIR):
        print(f"🗑  Deleting existing database at '{CHROMA_DB_DIR}'")
        shutil.rmtree(CHROMA_DB_DIR)

    if os.path.exists(CHROMA_DB_DIR):
        print(f"ℹ  Database already exists at '{CHROMA_DB_DIR}'. Use --reset to rebuild.")
        return

    # Step 1 — Parse PDFs
    documents = load_pdfs(args.source)
    if not documents:
        print("❌ No documents to process. Exiting.")
        return

    # Step 2 — Chunk
    chunks = split_documents(documents)

    # Step 3 — Embed & store
    create_vector_store(chunks, CHROMA_DB_DIR)

    print("\n🎉 Done! Vector database is ready at:", CHROMA_DB_DIR)


if __name__ == "__main__":
    main()
