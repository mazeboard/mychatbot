import React, { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [confidence, setConfidence] = useState(null);
  const [ingestStatus, setIngestStatus] = useState("");

  const ask = async () => {
    setAnswer(null);
    try {
      const res = await axios.post(
        `/search`,
        { question },
        { headers: { "Content-Type": "application/json" } }
      );
      setAnswer(res.data.answer);
      setSources(res.data.sources || []);
      setConfidence(res.data.confidence);
    } catch (err) {
      console.error(err);
      setAnswer("Error contacting orchestrator");
    }
  };

  return (
    <div style={{ padding: 30, fontFamily: "Arial, sans-serif" }}>
      <h2>Mini RAG Assistant (POC)</h2>

      <div style={{ marginBottom: 10 }}>
        <input
          style={{ width: "60%", padding: 8 }}
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={ask} style={{ marginLeft: 8 }}>
          Ask
        </button>
      </div>

      {answer && (
        <div style={{ marginTop: 20 }}>
          <h3>Answer</h3>
          <div
            style={{
              whiteSpace: "pre-wrap",
              border: "1px solid #ddd",
              padding: 12,
            }}
          >
            {answer}
          </div>
          <div style={{ marginTop: 10 }}>
            <strong>Confidence:</strong> {confidence}
          </div>

          <div style={{ marginTop: 10 }}>
            <h4>Sources</h4>
            <ul>
              {sources.map((s, i) => (
                <li key={i}>
                  <strong>
                    ({s.score}) {s.url}
                  </strong>{" "}
                  — {s.content}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
