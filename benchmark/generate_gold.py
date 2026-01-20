import json, random, os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Load configuration from environment variables
INPUT_FILE = os.getenv("INPUT_FILE", "data/output/ec2-ug_recursive_small.json")
OUTPUT_GOLD_FILE = os.getenv("OUTPUT_GOLD_FILE", "benchmark/gold/ec2-ug_gold_responses.json")
NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS", "50"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))


class QAPair(BaseModel):
    question : str = Field(..., description="A specific technical question based on the text")
    answer : str = Field(..., description="The concise answer to the question based ONLY on the text")

def setup_chain():
    """Create and return a LangChain chain for generating Q&A pairs from text chunks.
    
    The chain pipeline:
    1. Takes a document chunk as input
    2. Sends it to GPT-4.1-mini with a technical examiner prompt
    3. Parses the response into a structured QAPair object
    
    Returns:
        LangChain Runnable: A chain that accepts {"text_chunk": str} and returns QAPair
    """
    # Initialize LLM with GPT-4.1-mini for diverse, high-quality question generation
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    
    # Parser ensures consistent JSON output matching QAPair schema
    parser = JsonOutputParser(pydantic_object=QAPair)

    # Prompt template guides LLM to generate technical questions answerable only from chunk
    template = """
    You are a technical examiner creating a test for a Retrieval Augmented Generation (RAG) system.
    
    Given the following text chunk from a technical manual, generate a specific technical question 
    that can be answered using ONLY this text. Then provide the correct answer.

    Text Chunk:
    {text_chunk}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["text_chunk"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Pipe operator chains: prompt -> LLM -> parser
    chain = prompt | llm | parser

    return chain


def main():
    """Main execution function to generate gold standard Q&A pairs.
    
    Process:
    1. Load chunked documents from ingestion output (data/output/*.json)
    2. Filter for viable chunks (min 50 words) to ensure meaningful content
    3. Randomly sample NUM_QUESTIONS chunks for diversity
    4. Generate Q&A pairs using LLM for each chunk
    5. Save gold dataset with source metadata for RAGAS evaluation
    
    The gold dataset tracks source_chunk_id to calculate Hit Rate:
    - If RAG retrieves the correct source chunk, Hit Rate += 1
    - If RAG retrieves wrong chunk, Hit Rate += 0
    
    Raises:
        FileNotFoundError: If INPUT_FILE doesn't exist (run /ingest "INPUT_FILE" is not definedfirst)
    """
    # Verify input file exists from ingestion pipeline
    if not os.path.exists(INPUT_FILE):
        print(f"Could not find {INPUT_FILE}, please run the ingestion first.")
        return
    
    # Load pre-chunked document JSON from ingestion
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Filter chunks with >50 words to ensure sufficient content for Q&A generation
    viable_chunks = [
        chunk for chunk in chunks
        if len(chunk["content"].strip().split()) > 50
    ]

    if not viable_chunks:
        print("No viable chunks found for question generation.")
        return
    
    # Randomly sample chunks to ensure dataset diversity
    selected_chunks = random.sample(
        viable_chunks,
        min(NUM_QUESTIONS, len(viable_chunks))
    )

    print(
        f"Generating {len(selected_chunks)} Q&A pairs "
        f"from {len(viable_chunks)} viable chunks..."
    )

    # Initialize the Q&A generation chain
    qa_chain = setup_chain()
    golden_data = []

    # 1. Prepare the list of inputs for the batch
    batch_inputs = [{"text_chunk": chunk["content"]} for chunk in selected_chunks]

    print("Running batch generation (this may take a moment)...")

    # 2. Execute in parallel
    # return_exceptions=True ensures one failure doesn't crash the whole script
    batch_results = qa_chain.batch(batch_inputs, config={"max_concurrency": MAX_CONCURRENCY}, return_exceptions=True)

    # 3. Process batch results and build golden_data
    for i, (chunk, result) in enumerate(zip(selected_chunks, batch_results)):
        # Skip failed results (exceptions)
        if isinstance(result, Exception):
            print(f"Error generating Q&A for chunk {i}: {result}")
            continue
        
        try:
            # Build gold entry with metadata for RAGAS evaluation
            entry = {
                "id": str(i),
                "source_chunk_id": chunk.get("chunk_id"),  # For Hit Rate calculation
                "question": result["question"],
                "ground_truth": result["answer"],
                "source_text": chunk["content"]  # Reference for manual inspection
            }
            golden_data.append(entry)
            print(f"Generated Q&A pair {i+1}/{len(selected_chunks)}")
        except Exception as e:
            print(f"Error processing result for chunk {i}: {e}")
            continue

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_GOLD_FILE), exist_ok=True)
    
    # Write gold dataset to JSON for benchmarking
    with open(OUTPUT_GOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(golden_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(golden_data)} Q&A pairs to {OUTPUT_GOLD_FILE}")

if __name__ == "__main__":
    """Entry point: Generate gold standard dataset for RAG benchmarking."""
    main()