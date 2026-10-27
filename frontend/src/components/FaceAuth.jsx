import { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';

export default function FaceAuth({ onAuthenticated }) {
  const videoRef = useRef();
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [status, setStatus] = useState("Chargement des modèles d'IA...");

  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = '/models';
      try {
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        setIsModelLoaded(true);
        setStatus("Modèles prêts. Caméra en cours d'activation...");
        startVideo();
      } catch (error) {
        setStatus("Erreur de chargement des modèles : " + error.message);
      }
    };
    loadModels();
  }, []);

  const startVideo = () => {
    navigator.mediaDevices.getUserMedia({ video: {} })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setStatus("Regardez la caméra et cliquez sur 'Se connecter'");
        }
      })
      .catch((err) => setStatus("Erreur caméra : " + err.message));
  };

  const handleLogin = async () => {
    if (!videoRef.current) return;
    setStatus("Analyse du visage...");

    try {
      // Détection du visage
      const detection = await faceapi
        .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        setStatus("Visage détecté ! Authentification réussie.");
        // Arrêter la caméra
        const stream = videoRef.current.srcObject;
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
        }
        // Attendre 1 seconde pour voir le message, puis entrer
        setTimeout(() => onAuthenticated(true), 1000);
      } else {
        setStatus("Aucun visage détecté. Veuillez réessayer.");
      }
    } catch (error) {
      setStatus("Erreur d'analyse : " + error.message);
    }
  };

  return (
    <div style={{ 
      display: 'flex', flexDirection: 'column', alignItems: 'center', 
      padding: '30px', background: '#ffffff', borderRadius: '15px', 
      boxShadow: '0 10px 25px rgba(0,0,0,0.2)', maxWidth: '400px', margin: '50px auto' 
    }}>
      <h2 style={{ color: '#1e3a8a', marginBottom: '10px' }}>🔒 Authentification Médecin</h2>
      <p style={{ color: '#4b5563', marginBottom: '20px', textAlign: 'center', fontWeight: 'bold' }}>
        {status}
      </p>
      
      <div style={{ 
        position: 'relative', width: '320px', height: '240px', 
        background: '#000', borderRadius: '12px', overflow: 'hidden', 
        marginBottom: '25px', border: '4px solid #10b981' 
      }}>
        <video 
          ref={videoRef} 
          autoPlay 
          muted 
          style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
        />
      </div>

      <button 
        onClick={handleLogin} 
        disabled={!isModelLoaded}
        style={{ 
          padding: '15px 30px', background: isModelLoaded ? '#10b981' : '#9ca3af', 
          color: 'white', border: 'none', borderRadius: '8px', 
          fontSize: '16px', cursor: isModelLoaded ? 'pointer' : 'not-allowed', 
          fontWeight: 'bold', width: '100%', transition: 'background 0.3s'
        }}
      >
        {isModelLoaded ? "Se connecter" : "Chargement..."}
      </button>
    </div>
  );
}