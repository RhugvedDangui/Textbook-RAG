"""
RAG application — retrieves textbook content from ChromaDB and summarizes it using Ollama.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model:   ollama pull llama3
    3. Set SEARCH_QUERY below, then run: python app.py
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── ✏️  PUT YOUR SEARCH TOPIC HERE ───────────────────────────────────────────
SEARCH_QUERY = "\"primary_topic\": \"Spiral Model\", \"sub_topic\": \"Introduction\", \"speaker\": \"Vishwari Shali\", \"verbatim_text\": \"In today's session, we will discuss about the next important model that is spiral model. Let's start the session. In today's session, we will discuss about introduction phases, when to use spiral model and their advantages and disadvantages. Let's see all these points one by one.\""
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "textbook_rag"
OLLAMA_MODEL = "mistral:latest"   # change to any model you have pulled, e.g. "mistral"
TOP_K = 5                 # number of chunks to retrieve

SYSTEM_PROMPT = """\
You are a helpful teaching assistant. Your role is to summarize and explain \
the provided lecture transcript segment so students can understand and revise the topic.

Use ONLY the lecture content provided in the context below. \
If the context does not contain enough information to answer, say so clearly — do not add outside information.

Provide a clear, well-structured summary. Use headings for the main topic and bullet points for key details.

Context from the lecture:
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


# ── Search & Display ──────────────────────────────────────────────────────────

def run(query: str) -> None:
    print(f"\n🔍 Query: {query}\n")

    db = get_vector_store()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)

    # ── Section 1: Raw textbook content ──────────────────────────────────────
    print("━" * 70)
    print("📖  TEXTBOOK CONTENT")
    print("━" * 70)
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        print(f"\n[{i}] {source}  |  Page {page}")
        print("─" * 70)
        print(doc.page_content.strip())
    print()

    # ── Section 2: LLM summary ───────────────────────────────────────────────
    print("━" * 70)
    print(f"🤖  AI SUMMARY  ({OLLAMA_MODEL})")
    print("━" * 70)
    print("⏳ Generating summary...\n")

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Topic / Question: {question}\n\nSummarize this topic based on the textbook content above."),
    ])

    def format_docs(doc_list):
        parts = []
        for i, doc in enumerate(doc_list, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    chain = (
        {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)
    print(answer)
    print("━" * 70)



# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(SEARCH_QUERY)
