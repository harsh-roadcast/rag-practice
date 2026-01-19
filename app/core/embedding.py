from langchain_community.embeddings import SentenceTransformerEmbeddings

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._sentence_transformer_small = None 
        self._sentence_transformer_medium = None 
        self._sentence_transformer_large = None  
    @property
    def sentence_transformer_small(self) -> SentenceTransformerEmbeddings:
        if self._sentence_transformer_small is None:
            self._sentence_transformer_small = SentenceTransformerEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._sentence_transformer_small
    
    @property
    def sentence_transformer_medium(self) -> SentenceTransformerEmbeddings:
        if self._sentence_transformer_medium is None:
            self._sentence_transformer_medium = SentenceTransformerEmbeddings(
                model_name="BAAI/bge-base-en-v1.5"
            )
        return self._sentence_transformer_medium
    
    @property
    def sentence_transformer_large(self) -> SentenceTransformerEmbeddings:
        if self._sentence_transformer_large is None:
            self._sentence_transformer_large = SentenceTransformerEmbeddings(
                model_name="BAAI/bge-large-en-v1.5"
            )
        return self._sentence_transformer_large
    
embeddings_base = Embedder()