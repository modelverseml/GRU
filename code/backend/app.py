"""
FastAPI backend for the vanilla-RNN sentiment app.

Loads the trained PyTorch + TensorFlow models for all four encoders and exposes
a /predict endpoint that the React frontend calls with a piece of text. It
returns the predicted sentiment and class confidences for every model.

Run it (from this folder, with the project venv active):

    uvicorn app:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm up the encoders + models once at startup so the first request is fast
    print("warming up encoders and models...")
    info = predictor.warmup()
    app.state.info = info
    print(f"ready: encoders={info['encoders']} labels={info['labels']}")
    yield


app = FastAPI(title="Vanilla-RNN Sentiment API", lifespan=lifespan)

# allow the vite dev server (and anything else, for local dev) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    # what encoders / labels / frameworks the frontend can expect
    return app.state.info


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        return {"text": text, "results": []}
    return {"text": text, "results": predictor.predict(text)}
