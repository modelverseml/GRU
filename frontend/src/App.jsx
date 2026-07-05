import { useEffect, useState } from "react";

// in dev we hit the vite proxy at /api (-> FastAPI on :8000). override with
// VITE_API_URL when the backend lives somewhere else (e.g. a deployed host).
const API_BASE = import.meta.env.VITE_API_URL || "/api";

// fixed display order for the encoders and the sentiment classes
const ENCODER_ORDER = ["word2vec", "fasttext", "glove", "bert"];
const LABEL_ORDER = ["negative", "neutral", "positive"];

// per-sentiment styling (color + emoji)
const SENTIMENT = {
  positive: { color: "#15803d", soft: "#dcfce7", emoji: "😊" },
  neutral: { color: "#b45309", soft: "#fef3c7", emoji: "😐" },
  negative: { color: "#b91c1c", soft: "#fee2e2", emoji: "😞" },
};

// a short descriptor for each encoder, shown under its name
const ENCODER_META = {
  word2vec: "Google News · 300d",
  fasttext: "Wiki-news · 300d",
  glove: "Wikipedia · 300d",
  bert: "Transformer · 768d",
};

const FRAMEWORK_ICON = { pytorch: "🔥", tensorflow: "🧠", manual: "🧮" };
const FRAMEWORK_NAME = { pytorch: "PyTorch", tensorflow: "TensorFlow", manual: "from-scratch NumPy" };
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

const SAMPLES = [
  "this product is absolutely amazing, i love it",
  "complete waste of money, broke after one day",
  "it works fine, nothing special though",
];

function ProbBar({ label, value }) {
  const { color } = SENTIMENT[label];
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-track">
        <div
          className="prob-fill"
          style={{
            width: `${value * 100}%`,
            background: `linear-gradient(90deg, ${color}99, ${color})`,
          }}
        />
      </div>
      <span className="prob-pct">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function ModelCard({ result }) {
  const s = SENTIMENT[result.label];
  return (
    <div className="model-card">
      <div className="model-head">
        <span className="framework">
          {FRAMEWORK_ICON[result.framework]} {result.framework}
        </span>
        <span
          className="pred-label"
          style={{ background: s.soft, color: s.color }}
        >
          {s.emoji} {result.label}
        </span>
      </div>
      <div className="confidence" style={{ color: s.color }}>
        {(result.confidence * 100).toFixed(1)}
        <span className="confidence-pct">%</span>
        <span className="confidence-cap">confidence</span>
      </div>
      <div className="probs">
        {LABEL_ORDER.map((lbl) => (
          <ProbBar key={lbl} label={lbl} value={result.probabilities[lbl] ?? 0} />
        ))}
      </div>
    </div>
  );
}

function Consensus({ results }) {
  // majority vote across all model predictions
  const tally = {};
  results.forEach((r) => (tally[r.label] = (tally[r.label] || 0) + 1));
  const [winner, count] = Object.entries(tally).sort((a, b) => b[1] - a[1])[0];
  const s = SENTIMENT[winner];
  return (
    <div className="consensus" style={{ background: s.soft }}>
      <span className="consensus-emoji">{s.emoji}</span>
      <div>
        <div className="consensus-label" style={{ color: s.color }}>
          {winner}
        </div>
        <div className="consensus-sub">
          {count} of {results.length} models agree
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState(SAMPLES[0]);
  const [results, setResults] = useState([]);
  const [submitted, setSubmitted] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState(null);   // { encoders, frameworks } actually loaded

  useEffect(() => {
    fetch(`${API_BASE}/info`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setInfo(null));
  }, []);

  async function runPredict() {
    const trimmed = text.trim();
    if (!trimmed) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });
      if (!res.ok) throw new Error(`server returned ${res.status}`);
      const data = await res.json();
      setResults(data.results || []);
      setSubmitted(data.text || trimmed);
    } catch (e) {
      setError(`Could not reach the backend (${e.message}). Is it running on :8000?`);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runPredict();
  }

  const byEncoder = ENCODER_ORDER.map((enc) => ({
    encoder: enc,
    rows: results.filter((r) => r.encoder === enc),
  })).filter((g) => g.rows.length > 0);

  // describe only what the backend actually loaded (no over-claiming)
  const subtitle = info && info.encoders && info.frameworks
    ? `One review, ${plural(info.encoders.length, "embedding")} × ${plural(info.frameworks.length, "implementation")} of the same GRU (${info.frameworks.map((f) => FRAMEWORK_NAME[f] || f).join(", ")}). Watch them compare.`
    : "One review, the same GRU built several ways. Watch them compare.";

  return (
    <div className="page">
      <header className="hero">
        <h1>
          <span className="accent">GRU</span> Sentiment
        </h1>
        <p className="sub">{subtitle}</p>
      </header>

      <section className="input-card">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a product review…"
          rows={3}
        />
        <div className="controls">
          <button className="predict-btn" onClick={runPredict} disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" /> Predicting…
              </>
            ) : (
              "Analyze sentiment"
            )}
          </button>
          <span className="hint">⌘/Ctrl + Enter</span>
          <div className="samples">
            {SAMPLES.map((s, i) => (
              <button key={i} className="sample" onClick={() => setText(s)}>
                Sample {i + 1}
              </button>
            ))}
          </div>
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      {submitted && !error && results.length > 0 && (
        <section className="results fade-in">
          <div className="results-top">
            <p className="submitted">
              “{submitted}”
            </p>
            <Consensus results={results} />
          </div>

          {byEncoder.map((group) => (
            <div key={group.encoder} className="encoder-block">
              <div className="encoder-head">
                <h2>{group.encoder}</h2>
                <span className="encoder-meta">{ENCODER_META[group.encoder]}</span>
              </div>
              <div className="cards">
                {group.rows.map((r) => (
                  <ModelCard key={`${r.encoder}-${r.framework}`} result={r} />
                ))}
              </div>
            </div>
          ))}
        </section>
      )}

      {!submitted && !error && (
        <p className="empty-hint">Enter a review above to see the models compare.</p>
      )}
    </div>
  );
}
