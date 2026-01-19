import json
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

try:
    from app.core.embedding import embeddings_base
except ModuleNotFoundError:  # fallback for running as a script inside app/core
    from embedding import embeddings_base  # type: ignore


class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.embedder = embeddings_base

    def chunk_text_recursive(self, documents: List[Document],chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """
        Chunks text using RecursiveCharacterTextSplitter.
        Args:
            documents (List[Document]): List of documents to be chunked.
            chunk_size (int): Size of each chunk.
            chunk_overlap (int): Overlap between chunks.
        Returns:
            List[Document]: List of chunked documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(documents)

    def chunk_text_length_based(self, documents: List[Document],chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """
        Chunks text using CharacterTextSplitter based on length.
        Args:
            documents (List[Document]): List of documents to be chunked.
            chunk_size (int): Size of each chunk.
            chunk_overlap (int): Overlap between chunks.
        Returns:
            List[Document]: List of chunked documents.
        """
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(documents)
    
    def chunk_text_semantic(self, documents: List[Document], embedding_type: str = "medium") -> List[Document]:

        """
        Chunks text using SemanticChunker.
        Args:
            documents (List[Document]): List of documents to be chunked.
            embedding_type (str): Type of embedding to use ("small", "medium", "large").
        Returns:
            List[Document]: List of chunked documents.
        """
        
        if embedding_type == "small":
            embeddings = self.embedder._sentence_transformer_small
        elif embedding_type == "medium":
            embeddings = self.embedder._sentence_transformer_medium
        else:
            embeddings = self.embedder._sentence_transformer_large
        
        if embeddings is None:
            raise ValueError(f"Embedding model for type {embedding_type} is not initialized.")
        
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        return splitter.split_documents(documents)
    

chunker_service = Chunker()
