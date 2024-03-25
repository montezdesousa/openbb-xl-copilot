"""Embeddings module."""
import json
import os
from typing import Any, Dict, List, Optional
import requests
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

XL_FUNCS_URL = "https://excel.openbb.co/assets/functions.json"
XL_DOCS_PATH = "reference_map.json"
XL_EMBS_PATH = "db"


class Embeddings:
    SEARCH_KWARGS: int = 1  # Number of results to return.
    MAX_TOKENS: int = 1000  # Maximum number of tokens to use for context.

    model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    def __init__(self) -> None:
        self.documents = self.create_documents()
        self.db: Optional[FAISS] = None

    def create_documents(self) -> List[Document]:
        # Batch this if it takes too long.
        return [
            Document(
                page_content=json.dumps(func),
                metadata={"index": i, "function": func[1]["signature"]},
            )
            for i, func in enumerate(self.read_xl_docs().items())
        ]

    def train(self) -> None:
        self.db = FAISS.from_documents(
            self.documents,
            self.model,
        )

    def get_retriever(self) -> VectorStoreRetriever:
        if self.db is None:
            raise ValueError("No vector store found.")
        return self.db.as_retriever(
            search_kwargs={"k": self.SEARCH_KWARGS},
            max_tokens_for_context=self.MAX_TOKENS,
        )

    def save(self) -> None:
        if self.db is None:
            raise ValueError("No vector store found.")
        self.db.save_local(XL_EMBS_PATH)

    def load(self) -> None:
        if not os.path.exists(XL_EMBS_PATH):
            raise ValueError("No vector store found.")
        self.db = FAISS.load_local(XL_EMBS_PATH, self.model)

    @staticmethod
    def fetch_xl_funcs():
        r = requests.get(XL_FUNCS_URL, timeout=10)
        with open(XL_DOCS_PATH, "w") as f:
            json.dump(r.json(), f, indent=2)

    @staticmethod
    def read_xl_docs() -> Dict[str, Any]:
        with open(XL_DOCS_PATH) as f:
            docs = json.load(f)
        return docs
