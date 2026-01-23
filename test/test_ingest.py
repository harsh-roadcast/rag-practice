from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from app.core.chunking import chunker_service

# 1. Setup Path
PDF_PATH = Path("data/manuals/aws-overview.pdf") # Make sure file exists

def main():
    print(f"--- Loading {PDF_PATH} ---")
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 1. Test Length Based
    print("\n--- Testing Length-Based Chunking ---")
    chunks_len = chunker_service.chunk_text_length_based(docs)
    print(f"Generated {len(chunks_len)} chunks.")
    print(f"Sample: {chunks_len[0].page_content[:100]}...")

    # 2. Test Recursive
    print("\n--- Testing Recursive Chunking ---")
    chunks_rec = chunker_service.chunk_text_recursive(docs)
    print(f"Generated {len(chunks_rec)} chunks.")
    print(f"Sample: {chunks_rec[0].page_content[:100]}...")

    # 3. Test Semantic (This will be slow!)
    print("\n--- Testing Semantic Chunking (Medium) ---")
    chunks_sem = chunker_service.chunk_text_semantic(docs, embedding_type="medium")
    print(f"Generated {len(chunks_sem)} chunks.")

if __name__ == "__main__":
    main()