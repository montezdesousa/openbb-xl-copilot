"""FastAPI server for the chatbot."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bot import Bot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    query: str


class Answer(BaseModel):
    content: str


chat_bot = Bot.create()


@app.post("/query")
async def ask(question: Question) -> Answer:
    """Ask the bot a question."""
    try:
        answer = Answer(content=chat_bot.ask(question.query))
    except Exception as e:
        answer = Answer(content=str(e))

    return answer


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", reload=True)
