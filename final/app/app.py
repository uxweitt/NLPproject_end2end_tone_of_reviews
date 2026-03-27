from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from ml.model import load_model

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    
    yield
    
app = FastAPI(lifespan=lifespan)

class  SentimentResponse(BaseModel):
    text: str
    sentiment_label: str
    sentiment_score: float

@app.get("/")
def index():
    return {"text": "Sentiment Analysis"}
    

@app.get("/predict")
def predict_sentiment(text):
    sentiment = model(text)
    response = SentimentResponse(
        text=text,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
    )
    return response