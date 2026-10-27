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

  // --- ÉTATS POUR LE COMPTE-RENDU (NOUVEAU) ---
  const [report, setReport] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);

  // --- FONCTION D'UPLOAD ---
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    setResult(null);
    setReport(null); // Réinitialiser le rapport à chaque nouvelle image

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

  // --- FONCTION POUR GÉNÉRER LE COMPTE-RENDU (NOUVEAU) ---
  const handleGenerateReport = async () => {
    if (!result) return;
    setLoadingReport(true);
    try {
      const response = await fetch(`${API_BASE}/generate-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          polyp_area_ratio: result.polyp_area_ratio,
          original_width: result.original_width,
          original_height: result.original_height
        }),
      });
      const data = await response.json();
      setReport(data.report);
    } catch (err) {
      console.error(err);
      setError("Erreur lors de la génération du compte-rendu.");
    } finally {
      setLoadingReport(false);
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

      {/* --- BOUTON ET AFFICHAGE DU COMPTE-RENDU (NOUVEAU) --- */}
      {result && (
        <div style={{ marginTop: '20px', textAlign: 'center' }}>
          <button 
            onClick={handleGenerateReport} 
            disabled={loadingReport}
            style={{ padding: '12px 24px', background: '#8b5cf6', color: 'white', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 'bold', fontSize: '16px' }}
          >
            {loadingReport ? "Génération en cours (IA)..." : "📄 Générer le compte-rendu IA"}
          </button>
        </div>
      )}

      {report && (
  <div style={{ 
    marginTop: '20px', 
    padding: '20px', 
    background: '#f3f4f6', 
    borderRadius: '8px', 
    borderLeft: '4px solid #8b5cf6', 
    color: '#000000', 
    maxWidth: '800px', 
    margin: '20px auto',
    maxHeight: '400px',       // NOUVEAU : Limite la hauteur à 400px
    overflowY: 'auto'         // NOUVEAU : Ajoute la barre de défilement
  }}>
    <h3 style={{ marginTop: 0, color: '#4c1d95', borderBottom: '1px solid #ddd', paddingBottom: '10px' }}>
      Compte-rendu Clinique
    </h3>
    <p style={{ whiteSpace: 'pre-line', fontSize: '15px', lineHeight: '1.6' }}>{report}</p>
  </div>
)}
      
      <ChatBot />
    </div>
  );
}

export default App;