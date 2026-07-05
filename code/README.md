# Code layout

```
code/
├── model_building/   # everything that produces the models
│   ├── data_generation.py             # 1. download + split reviews -> data/raw/
│   ├── encoder.py                     # 2. build+trim encoders + encode splits
│   ├── model_artifacts_generation.py  # 3. train the RNNs -> data/model_artifacts/
│   ├── manual_rnn.py                  #    from-scratch NumPy RNN (used by step 3)
│   └── run_pipeline.py                #    runs steps 1-3 end to end
└── backend/          # serves the trained models to the frontend
    ├── predictor.py                   # framework-agnostic: text -> per-model predictions
    ├── app.py                         # FastAPI wrapper exposing /predict
    └── requirements.txt
```

The React UI lives at the repo root in `../frontend/`.

## 1. Build everything (one command)

```bash
cd code/model_building
python run_pipeline.py                  # full run: data -> encoder -> model
# or reuse existing data + encoders and just retrain:
python run_pipeline.py --skip data encoder
```

This produces the 12 artifacts in `data/model_artifacts/` — three models per
encoder: `pytorch_*.pt`, `tensorflow_*.keras`, and `manual_*.npz` (a vanilla RNN
written from scratch in NumPy with its own BPTT + Adam, no deep-learning lib).

### How the encoders stay small

`encoder.py` downloads each pretrained encoder, **trims it to our dataset's
vocabulary in memory, and saves only the small copy** — the full gigabyte files
are never written to disk. BERT (`bert-base-uncased`) loads straight from
HuggingFace at runtime, so its weights aren't stored in the repo either.
Trimming is lossless for our data (every dataset word is kept; dropped words were
out-of-vocab anyway, mapping to zeros exactly as before).

What's committed vs ignored (see root `.gitignore`):

| committed | ignored (regenerable) |
|---|---|
| `data/text_encoder_small/` (~75 MB) | `data/text_embedding/` (GBs, training only) |
| `data/model_artifacts/` (~14 MB) | (full encoders are never written to disk) |
| `data/raw/` (~9 MB) | |

## 2. Run the backend API

```bash
cd code/backend
pip install -r requirements.txt              # fastapi + uvicorn (ML deps come from the project venv)
uvicorn app:app --reload --port 8000
```

It warms up all encoders + models at startup, then serves:

- `POST /predict`  `{ "text": "..." }` → predictions for every (encoder, framework)
- `GET  /health`, `GET /info`

## 3. Run the frontend

```bash
cd frontend
npm install
npm run dev                                   # http://localhost:5173
```

The dev server proxies `/api/*` to the backend on `:8000`, so start the backend
first. Type a review, hit **Predict**, and see word2vec / fasttext / glove / bert
side by side, each with its PyTorch, TensorFlow, and from-scratch NumPy
prediction + confidences, plus a consensus vote across all models.

## Note on Streamlit

`backend/predictor.py` has no web framework in it — a future Streamlit app can
`import predictor` and call `predictor.predict(text)` directly, reusing the exact
same model-loading logic.
