from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class LLMService:
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.1):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        system_template = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the answer is not in the context, say \"I don't know based on the provided documents.\" "
            "Keep the answer concise and professional."
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _format_context(self, context_chunks: List[Dict[str, str]]) -> str:
        """Create a readable context block for the LLM."""
        parts = []
        for chunk in context_chunks:
            text = None
            metadata = None

            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content")
                metadata = chunk.get("metadata")
            else:
                text = getattr(chunk, "text", None) or getattr(chunk, "content", None) or getattr(chunk, "page_content", None)
                metadata = getattr(chunk, "metadata", None)

            if not text:
                continue

            if metadata:
                parts.append(f"{text}\n[metadata: {metadata}]")
            else:
                parts.append(text)

        if not parts:
            raise ValueError("Cannot find usable text inside context chunks.")
        return "\n\n---\n\n".join(parts)

    def get_answer(self, query: str, context_chunks: List[Dict[str, str]]) -> str:
        """Return the model's answer for the query scoped to the provided chunks."""
        if not context_chunks:
            raise ValueError("Cannot find relevant context for the query.")

        context_text = self._format_context(context_chunks)
        return self.chain.invoke({"context": context_text, "question": query})

llm_service = LLMService()
