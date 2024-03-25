"""Bot module."""
import os

# import json

from datetime import datetime
from typing import Optional

from dotenv import dotenv_values
import instructor
from openai import OpenAI
from langchain_core.vectorstores import VectorStoreRetriever
from embeddings import Embeddings
import pytz
from models import FunctionMessage

os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_date = datetime.now().astimezone(pytz.timezone("America/New_York"))

PROMPT = f"""
    You are an experienced financial analyst that can access the documentation of OpenBB Add-in for Excel
    and accurately create Excel commands to retrieve the required OpenBB data.

    Current date: {current_date.strftime("%Y-%m-%d")}
    Current day of the week: {current_date.strftime("%A")}

    From the information below, provide the relevant Excel commands for users' questions. If relevant, use the examples as guidance.

    Using the information provided from the description, parameters and data field, determine the most appropriate functions to satisfy the user's request.
    In case users' request include any specific query parameters, please make sure to include them in the function similar to the examples provided in the documentation.

    If no specific symbol is provided, use AAPL as the default symbol.

    You are able to express yourself purely through JSON, strictly and precisely adhering to the provided schemas. 

"""


class Bot:
    """Bot class."""

    def __init__(
        self, api_key: str, model: str, retriever: Optional[VectorStoreRetriever] = None
    ) -> None:
        self.llm = instructor.patch(OpenAI(api_key=api_key))
        self.model = model
        self.retriever = retriever
        self.PROMPT = PROMPT

    def get_context(self, question: str) -> str:
        """Get context."""
        if self.retriever:
            return str(self.retriever.get_relevant_documents(question))
        return ""

    def ask(self, question: str) -> str:
        """Ask a question."""
        context = self.get_context(question)

        result: FunctionMessage = self.llm.chat.completions.create(
            model=self.model,
            temperature=0,
            response_model=FunctionMessage,
            messages=[
                {
                    "role": "system",
                    "content": self.PROMPT,
                },
                {"role": "user", "content": f"{context}"},
                {"role": "user", "content": f"Question: {question}"},
            ],
            validation_context={"text_chunk": context},
        )

        return result.to_xl()

    @classmethod
    def create(cls) -> "Bot":
        config: dict = dotenv_values(".env")
        e = Embeddings()

        try:
            e.load()
            retriever = e.get_retriever()
        except Exception:
            e.train()
            e.save()
            retriever = e.get_retriever()

        return cls(
            api_key=config.get("OPENAI_API_KEY", ""),
            model=config.get("OPENAI_MODEL", ""),
            retriever=retriever,
        )
