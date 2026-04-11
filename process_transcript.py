import json
import nltk
import re
import ollama

# ─── CONFIGURATION ────────────────────────────────────────────────────────────────
# Recommended model: llama3:8b is a great balance of speed and accuracy for this task.
# You can also use other models like mistral:latest.
OLLAMA_MODEL_FOR_TOPICS = "llama3:8b" 
MIN_CHUNK_SIZE_WORDS = 100
MAX_CHUNK_SIZE_WORDS = 180
OUTPUT_FILENAME = "chunks_output_processed.txt"
# ───────────────────────────────────────────────────────────────────────────────────

# Download the 'punkt' tokenizer if you haven't already
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' model...")
    nltk.download('punkt')

TOPIC_SYSTEM_PROMPT = """
You are an expert at analyzing text and identifying its core subject matter.
Your task is to identify the main topic and a specific sub-topic from the given text.
The text is a segment from a longer educational lecture.

Rules:
1.  Read the text and determine the most accurate `primary_topic` (the overall subject, e.g., "Spiral Model", "Python Programming").
2.  Determine the most accurate `sub_topic` (a specific aspect of the primary topic, e.g., "Risk Analysis Phase", "For Loops").
3.  Your response MUST be a valid JSON object, and nothing else.
4.  Do not add any explanations or introductory text outside of the JSON object.

Example:
Input Text: "In this section, we'll discuss the advantages of the spiral model. The primary benefit is its handling of risk..."
Your Output:
{
  "primary_topic": "Spiral Model",
  "sub_topic": "Advantages"
}
"""

def get_topic_from_ollama(chunk_text: str) -> (str, str):
    """
    Uses an LLM to dynamically assign a primary and sub-topic to a chunk of text.
    """
    print(f"  > Getting topic for chunk...")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_FOR_TOPICS,
            messages=[
                {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
                {"role": "user", "content": chunk_text}
            ],
            format="json"
        )
        result_content = response['message']['content']
        parsed_json = json.loads(result_content)
        primary = parsed_json.get("primary_topic", "Unknown Topic")
        sub = parsed_json.get("sub_topic", "Unknown Sub-Topic")
        print(f"    > Topic: {primary} -> {sub}")
        return primary, sub
    except Exception as e:
        print(f"    > Error getting topic from Ollama: {e}")
        return "Error", "Could not determine topic"

def create_sentence_chunks(transcript: str, max_chunk_size: int):
    """
    Splits a transcript into sentence-based chunks with a hard word limit.
    Forcibly splits sentences if they lack punctuation and exceed the limit.
    """
    sentences = nltk.sent_tokenize(transcript)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence_words = sentence.split()
        
        # Fallback: If a single sentence is massive (e.g. missing periods), split it forcibly by words
        if len(sentence_words) > max_chunk_size:
            # Finalize current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                
            # Slice the huge sentence into chunks of `max_chunk_size`
            for i in range(0, len(sentence_words), max_chunk_size):
                sub_section = " ".join(sentence_words[i:i + max_chunk_size])
                # Only append if it's the exact max size, or keep the remainder as the start of the next chunk
                if len(sub_section.split()) == max_chunk_size:
                    chunks.append(sub_section)
                else:
                    current_chunk = sub_section
            continue

        # Normal logic for properly punctuated sentences
        if len((current_chunk + " " + sentence).split()) > max_chunk_size:
            # If the current chunk is not empty, finalize it
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start a new chunk with the current sentence
            current_chunk = sentence
        else:
            # Otherwise, add the sentence to the current chunk
            current_chunk += (" " + sentence if current_chunk else sentence)
            
    # Add the last remaining chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# ==========================================
# Execute the script
# ==========================================
if __name__ == "__main__":
    
    print("Starting transcript processing...")
    # Read the transcript text directly from our newly created file
    with open("transcript.txt", "r", encoding="utf-8") as f:
        raw_transcript = f.read()

    # 1. Split transcript into sentence groups
    text_chunks = create_sentence_chunks(raw_transcript, MAX_CHUNK_SIZE_WORDS)
    
    print(f"Split transcript into {len(text_chunks)} raw chunks.")
    
    # 2. Use LLM to assign topics to each chunk
    structured_chunks = []
    for i, chunk_text in enumerate(text_chunks):
        print(f"\nProcessing chunk {i+1}/{len(text_chunks)}...")
        primary_topic, sub_topic = get_topic_from_ollama(chunk_text)
        
        structured_chunks.append({
            "primary_topic": primary_topic,
            "sub_topic": sub_topic,
            "speaker": "Vishwari Shali",  # Assuming a single speaker for simplicity
            "verbatim_text": chunk_text
        })
        
    final_output = {"chunks": structured_chunks}
    
    # 3. Save the final output
    if final_output and "chunks" in final_output:
        print("\n=== SUCCESS: CHUNKS EXTRACTED AND PROCESSED ===\n")
        
        with open(OUTPUT_FILENAME, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        print(f"Successfully saved structured data to {OUTPUT_FILENAME}")

        for i, chunk in enumerate(final_output["chunks"]):
            print(f"Chunk {i+1}:")
            print(f"  Topic: {chunk.get('primary_topic')} -> {chunk.get('sub_topic')}")
            print(f"  Word Count: {len(chunk.get('verbatim_text', '').split())}")
    else:
        print("Failed to extract structured data.")
