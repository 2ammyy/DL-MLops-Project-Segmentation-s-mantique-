import { useState } from "react";
import MaskOverlay from "./MaskOverlay.jsx";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setResult(null);
    setLoading(true);
    setImage(URL.createObjectURL(file));

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Erreur HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>PolypCopilot</h1>
        <p>Segmentation de polypes colorectaux — Kvasir-SEG</p>
      </header>

      <label className="upload-btn">
        {loading ? "Inference en cours..." : "Choisir une image de coloscopie"}
        <input type="file" accept="image/*" onChange={handleUpload} disabled={loading} />
      </label>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="stats">
          <span>Modele : {result.model_version}</span>
          <span>Surface polype : {(result.polyp_area_ratio * 100).toFixed(1)}%</span>
          <span>Inference : {result.inference_ms} ms</span>
          <span>
            Image : {result.original_width}x{result.original_height}
          </span>
        </div>
      )}

      <MaskOverlay
        imageUrl={image}
        maskBase64={result ? result.mask_base64 : null}
      />
    </div>
  );
}
