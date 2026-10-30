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
    // --- APPLICATION PRINCIPALE ---
  return (
    <div style={{ background: '#f1f5f9', minHeight: '100vh', padding: '40px 20px', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ 
        maxWidth: '900px', margin: '0 auto', background: '#ffffff', 
        borderRadius: '16px', boxShadow: '0 4px 20px rgba(0,0,0,0.05)', 
        overflow: 'hidden', paddingBottom: '40px'
      }}>
        
        {/* Header de l'application */}
        <header style={{ background: '#1e293b', color: 'white', padding: '30px', textAlign: 'center' }}>
          <h1 style={{ margin: 0, fontSize: '28px', fontWeight: 'bold' }}>🩺 PolypCopilot</h1>
          <p style={{ margin: '5px 0 0 0', color: '#cbd5e1', fontSize: '14px' }}>
            Segmentation de polypes colorectaux — Kvasir-SEG
          </p>
        </header>

        <div style={{ padding: '30px' }}>
          {/* Bouton d'upload stylisé */}
          <div style={{ textAlign: 'center', marginBottom: '30px' }}>
            <label style={{ 
              display: 'inline-block', background: '#2563eb', color: 'white', 
              padding: '12px 24px', borderRadius: '8px', cursor: loading ? 'not-allowed' : 'pointer', 
              fontWeight: 'bold', transition: 'background 0.2s', boxShadow: '0 2px 5px rgba(37, 99, 235, 0.3)'
            }}>
              {loading ? "Inference en cours..." : "📁 Choisir une image de coloscopie"}
              <input type="file" accept="image/*" onChange={handleUpload} disabled={loading} style={{ display: 'none' }} />
            </label>
          </div>

          {error && <div style={{ background: '#fee2e2', color: '#991b1b', padding: '15px', borderRadius: '8px', textAlign: 'center', marginBottom: '20px', fontWeight: 'bold' }}>{error}</div>}

          {/* Statistiques (badges) */}
          {result && (
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'center', marginBottom: '30px' }}>
              <span style={{ background: '#f1f5f9', padding: '8px 16px', borderRadius: '20px', fontSize: '13px', color: '#334155', fontWeight: 'bold' }}>Modèle : {result.model_version}</span>
              <span style={{ background: '#f1f5f9', padding: '8px 16px', borderRadius: '20px', fontSize: '13px', color: '#334155', fontWeight: 'bold' }}>Surface : {(result.polyp_area_ratio * 100).toFixed(1)}%</span>
              <span style={{ background: '#f1f5f9', padding: '8px 16px', borderRadius: '20px', fontSize: '13px', color: '#334155', fontWeight: 'bold' }}>Temps : {result.inference_ms} ms</span>
            </div>
          )}

          {/* Overlay de l'image */}
          <div style={{ background: '#f8fafc', border: '2px dashed #e2e8f0', borderRadius: '12px', padding: '20px', minHeight: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {image ? (
               <MaskOverlay imageUrl={image} maskBase64={result ? result.mask_base64 : null} />
            ) : (
              <p style={{ color: '#94a3b8', fontWeight: 'bold' }}>Aucune image sélectionnée</p>
            )}
          </div>

          {/* Bouton Compte Rendu */}
          {result && (
            <div style={{ marginTop: '30px', textAlign: 'center', borderTop: '1px solid #e2e8f0', paddingTop: '20px' }}>
              <button 
                onClick={handleGenerateReport} 
                disabled={loadingReport}
                style={{ padding: '14px 28px', background: '#8b5cf6', color: 'white', borderRadius: '8px', border: 'none', cursor: loadingReport ? 'not-allowed' : 'pointer', fontWeight: 'bold', fontSize: '16px', boxShadow: '0 4px 10px rgba(139, 92, 246, 0.3)' }}
              >
                {loadingReport ? "Génération en cours (IA)..." : "📄 Générer le compte-rendu IA"}
              </button>
            </div>
          )}

          {/* Affichage du compte-rendu */}
          {report && (
            <div style={{ marginTop: '20px', padding: '25px', background: '#faf5ff', borderRadius: '12px', borderLeft: '4px solid #8b5cf6', color: '#000000', maxHeight: '400px', overflowY: 'auto' }}>
              <h3 style={{ marginTop: 0, color: '#4c1d95', borderBottom: '1px solid #e9d5ff', paddingBottom: '10px' }}>Compte-rendu Clinique</h3>
              <p style={{ whiteSpace: 'pre-line', fontSize: '15px', lineHeight: '1.7', color: '#1e293b' }}>{report}</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Le Chatbot flottant */}
      <ChatBot />
    </div>
  );
}

export default App;