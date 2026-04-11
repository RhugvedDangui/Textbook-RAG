"""
RAG application — retrieves textbook content from ChromaDB and summarizes it using Ollama.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model:   ollama pull llama3
    3. Set SEARCH_QUERY below, then run: python app.py
"""

import os
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "textbook_rag"
OLLAMA_MODEL = "mistral:latest"   # change to any model you have pulled, e.g. "mistral"
TOP_K = 3                 # number of chunks to retrieve per transcript section
INPUT_CHUNKS_FILE = "chunks_output_processed.txt"
OUTPUT_SUMMARY_FILE = "final_rag_summary.txt"

SYSTEM_PROMPT = """\
You are an expert teaching assistant. Your task is to summarize and explain a segment from a lecture transcript.
You have been provided with excerpts from a textbook as context to help ensure your explanation is accurate and comprehensive.

Use BOTH the lecture transcript segment AND the provided textbook context to write a rich, correct, and easy-to-understand summary.
Use formatting (bullet points, short paragraphs) to make it readable.

Textbook Context:
{context}
"""

# ── Load Vector Store ─────────────────────────────────────────────────────────

def get_vector_store() -> Chroma:
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(
            f"Vector database not found at '{CHROMA_DB_DIR}'.\n"
            "Run 'python create_vectordb.py' first to build it."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# ── Batch RAG Summarization ───────────────────────────────────────────────────

def format_docs(doc_list):
    parts = []
    for i, doc in enumerate(doc_list, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def run_rag_pipeline():
    # 1. Load Chroma
    print("Loading Vector Database...")
    db = get_vector_store()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    # 2. Setup LLM Chain
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Lecture Topic: {topic}\n\nLecture Transcript Segment:\n{transcript}\n\nPlease provide the summarized notes for this section."),
    ])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # 3. Load the chunks
    if not os.path.exists(INPUT_CHUNKS_FILE):
        print(f"Error: {INPUT_CHUNKS_FILE} not found. Please run chunking first.")
        return

    with open(INPUT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data.get("chunks", [])
    if not chunks:
        print("No chunks found in file.")
        return

    print(f"Found {len(chunks)} chunks. Beginning RAG-augmented sequential summarization...")
    
    final_output_parts = []

    # 4. Process each chunk
    for i, chunk in enumerate(chunks, 1):
        topic = f"{chunk.get('primary_topic')} - {chunk.get('sub_topic')}"
        transcript_text = chunk.get("verbatim_text", "")
        
        print(f"\n[{i}/{len(chunks)}] Processing: {topic}")
        
        # Retrieve context using the transcript text as the query
        docs = retriever.invoke(transcript_text)
        context_str = format_docs(docs)
        
        # Generate the summary
        print(f"   Generating summary with {OLLAMA_MODEL}...")
        answer = chain.invoke({
            "context": context_str,
            "topic": topic,
            "transcript": transcript_text
        })
        
        # Append to our final list
        section_header = f"### Section {i}: {topic}\n"
        final_output_parts.append(section_header + answer + "\n\n" + "-"*40 + "\n")

    # 5. Write everything to final file
    final_document = "\n".join(final_output_parts)
    with open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(final_document)
        
    print(f"\n✅ All chunks processed successfully! Full RAG summary saved to '{OUTPUT_SUMMARY_FILE}'")

if __name__ == "__main__":
    run_rag_pipeline()
