import { useState } from "react";
import MaskOverlay from "./MaskOverlay.jsx";
import ChatBot from './components/ChatBot';
import FaceAuth from './components/FaceAuth';
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

function App() {
  // --- ÉTATS POUR L'AUTHENTIFICATION ---
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // --- ÉTATS POUR L'APPLICATION ---
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- FONCTION D'UPLOAD ---
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Erreur lors de la prédiction");
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- ÉCRAN DE LOGIN ---
  if (!isAuthenticated) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: '#e5e7eb' }}>
        <FaceAuth onAuthenticated={setIsAuthenticated} />
      </div>
    );
  }

  // --- APPLICATION PRINCIPALE ---
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
      
      <ChatBot />
    </div>
  );
}

export default App;