import { useState } from 'react';

export default function ChatBot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

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
        body: JSON.stringify({ message: input }),
      });
      const data = await response.json();
      
      setMessages((prev) => [
        ...prev, 
        { role: 'assistant', text: data.answer, sources: data.sources }
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev, 
        { role: 'assistant', text: "Erreur de connexion au serveur." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      position: 'fixed', bottom: '20px', right: '20px', width: '350px', 
      background: '#ffffff', // Fond blanc forcé
      borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.25)', 
      display: 'flex', flexDirection: 'column', zIndex: 1000,
      border: '1px solid #ccc'
    }}>
      <div style={{ background: '#2563eb', color: '#ffffff', padding: '10px', borderTopLeftRadius: '10px', borderTopRightRadius: '10px', fontWeight: 'bold' }}>
        💬 Assistant Médical (RAG)
      </div>
      
      <div style={{ height: '300px', overflowY: 'auto', padding: '10px', display: 'flex', flexDirection: 'column', gap: '8px', background: '#f9f9f9' }}>
        {messages.length === 0 && (
          <p style={{ color: '#555', fontSize: '12px', textAlign: 'center' }}>
            Posez une question sur les polypes...
          </p>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} style={{ 
            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', 
            background: msg.role === 'user' ? '#dbeafe' : '#ffffff', // Bleu clair vs Blanc
            color: '#000000', // Texte noir forcé
            padding: '8px', borderRadius: '8px', maxWidth: '85%', fontSize: '13px',
            border: msg.role === 'assistant' ? '1px solid #e5e7eb' : 'none',
            boxShadow: msg.role === 'assistant' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none'
          }}>
            {msg.text}
            {msg.sources && msg.sources.length > 0 && (
              <div style={{ marginTop: '4px', fontSize: '10px', color: '#666', fontWeight: 'bold' }}>
                📚 Source : {msg.sources.join(', ')}
              </div>
            )}
          </div>
        ))}
        {loading && <div style={{ fontSize: '12px', color: '#2563eb', fontWeight: 'bold' }}>L'IA réfléchit...</div>}
      </div>

      <div style={{ display: 'flex', borderTop: '1px solid #e5e7eb' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Votre question..."
          style={{ flex: 1, padding: '10px', border: 'none', outline: 'none', borderBottomLeftRadius: '10px', color: '#000000', background: '#ffffff' }}
        />
        <button onClick={sendMessage} disabled={loading} style={{ padding: '10px 15px', background: '#2563eb', color: 'white', border: 'none', borderBottomRightRadius: '10px', cursor: 'pointer', fontWeight: 'bold' }}>
          Envoyer
        </button>
      </div>
    </div>
  );
}