import { useState } from 'react';

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lang, setLang] = useState('fr'); // État pour la langue ('fr' ou 'en')

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: input,
          language: lang // On envoie la langue choisie
        }),
      });
      const data = await response.json();
      
      setMessages((prev) => [
        ...prev, 
        { role: 'assistant', text: data.answer, sources: data.sources }
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev, 
        { role: 'assistant', text: lang === 'en' ? "Server connection error." : "Erreur de connexion au serveur." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) {
    return (
      <button 
        onClick={() => setIsOpen(true)}
        style={{ position: 'fixed', bottom: '30px', right: '30px', width: '60px', height: '60px', borderRadius: '50%', background: '#2563eb', color: 'white', border: 'none', fontSize: '28px', cursor: 'pointer', zIndex: 1000, boxShadow: '0 4px 12px rgba(37, 99, 235, 0.4)', display: 'flex', justifyContent: 'center', alignItems: 'center', transition: 'transform 0.2s' }}
        onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
        onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
      >
        💬
      </button>
    );
  }

  return (
    <div style={{ position: 'fixed', bottom: '30px', right: '30px', width: '380px', height: '500px', background: '#ffffff', borderRadius: '16px', boxShadow: '0 10px 25px rgba(0,0,0,0.2)', display: 'flex', flexDirection: 'column', zIndex: 1000, border: '1px solid #e5e7eb', overflow: 'hidden' }}>
      
      {/* Header avec bouton fermer et sélecteur de langue */}
      <div style={{ background: '#2563eb', color: '#ffffff', padding: '15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 'bold' }}>
          <span>{lang === 'en' ? '💬 Medical Assistant AI' : '💬 Assistant Médical IA'}</span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Sélecteur de langue */}
          <select 
            value={lang} 
            onChange={(e) => setLang(e.target.value)}
            style={{ background: '#1d4ed8', color: 'white', border: 'none', borderRadius: '4px', padding: '2px 5px', cursor: 'pointer', fontSize: '12px' }}
          >
            <option value="fr">🇫🇷 FR</option>
            <option value="en">🇬🇧 EN</option>
          </select>

          <button onClick={() => setIsOpen(false)} style={{ background: 'transparent', border: 'none', color: 'white', fontSize: '20px', cursor: 'pointer' }}>
            ✖
          </button>
        </div>
      </div>
      
      {/* Zone des messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '15px', display: 'flex', flexDirection: 'column', gap: '10px', background: '#f8fafc' }}>
        {messages.length === 0 && (
          <p style={{ color: '#64748b', fontSize: '14px', textAlign: 'center', marginTop: '20px' }}>
            {lang === 'en' ? 'Ask a question about polyps or medical guidelines...' : 'Posez une question sur les polypes ou les recommandations médicales...'}
          </p>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} style={{ alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', background: msg.role === 'user' ? '#dbeafe' : '#ffffff', color: '#000000', padding: '10px 14px', borderRadius: '12px', maxWidth: '85%', fontSize: '14px', border: msg.role === 'assistant' ? '1px solid #e2e8f0' : 'none', boxShadow: msg.role === 'assistant' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none' }}>
            {msg.text}
            {msg.sources && msg.sources.length > 0 && (
              <div style={{ marginTop: '6px', fontSize: '11px', color: '#64748b', fontWeight: 'bold' }}>
                📚 {lang === 'en' ? 'Source' : 'Source'} : {msg.sources.join(', ')}
              </div>
            )}
          </div>
        ))}
        {loading && <div style={{ fontSize: '13px', color: '#2563eb', fontWeight: 'bold' }}>{lang === 'en' ? 'AI is thinking...' : "L'IA réfléchit..."}</div>}
      </div>

      {/* Input */}
      <div style={{ display: 'flex', borderTop: '1px solid #e2e8f0', background: 'white' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder={lang === 'en' ? 'Your question...' : 'Votre question...'}
          style={{ flex: 1, padding: '15px', border: 'none', outline: 'none', color: '#000000', fontSize: '14px' }}
        />
        <button onClick={sendMessage} disabled={loading} style={{ padding: '15px 20px', background: '#2563eb', color: 'white', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}>
          {lang === 'en' ? 'Send' : 'Envoyer'}
        </button>
      </div>
    </div>
  );
}