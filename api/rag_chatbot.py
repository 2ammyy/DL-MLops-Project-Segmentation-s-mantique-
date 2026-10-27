import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline

# Configuration
CHROMA_PATH = "chroma_db"
GUIDELINES_DIR = Path("data/guidelines")

# 1. Initialisation des embeddings GRATUITS (SentenceTransformers)
# Utilise le modèle all-MiniLM-L6-v2 (~80MB, téléchargé automatiquement)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="medical_guidelines",
    embedding_function=embedding_fn
)

# 2. Initialisation du LLM GRATUIT (HuggingFace Transformers)
# Qwen2.5-0.5B est ultra-léger (~1GB) et tourne sur CPU
print("Chargement du modèle LLM (peut prendre 1-2 min au premier lancement)...")
llm_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",  # Change en "cuda" si tu as un GPU NVIDIA
    max_new_tokens=256
)
print("LLM prêt !")

def ingest_guidelines():
    """Charge les fichiers texte dans la base vectorielle."""
    if collection.count() > 0:
        print("Base de connaissances déjà chargée.")
        return

    documents = []
    metadatas = []
    ids = []

    for file_path in GUIDELINES_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            chunks = content.split("\n\n")
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append(chunk)
                    metadatas.append({"source": file_path.name})
                    ids.append(f"{file_path.stem}_{i}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"{len(documents)} chunks ajoutés à la base.")

def ask_question(question: str) -> dict:
    """Pose une question au chatbot en utilisant le RAG."""
    # 1. Recherche des documents pertinents
    results = collection.query(query_texts=[question], n_results=2)
    
    if not results['documents'] or not results['documents'][0]:
        return {"answer": "Je n'ai pas trouvé d'information pertinente.", "sources": []}

    context = "\n\n".join(results['documents'][0])
    sources = [meta['source'] for meta in results['metadatas'][0]]

    # 2. Génération avec le LLM local
    prompt = f"""<|im_start|>system
Tu es un assistant médical. Réponds UNIQUEMENT avec le contexte fourni.
<|im_end|>
<|im_start|>user
Contexte :
{context}

Question : {question}
<|im_end|>
<|im_start|>assistant
"""
    
    output = llm_pipeline(prompt, do_sample=False, temperature=0.1)
    answer = output[0]['generated_text'].split("<|im_start|>assistant\n")[-1].strip()
    
    return {
        "answer": answer,
        "sources": list(set(sources))
    }

# Ingestion au démarrage
ingest_guidelines()

def generate_report(polyp_area_ratio: float, original_width: int, original_height: int) -> str:
    """Génère un compte-rendu médical basé sur les résultats de la segmentation."""
    
    # Simulation d'une taille en mm basée sur le ratio de surface
    size_mm = (polyp_area_ratio * 100) 
    if size_mm < 5:
        risk = "faible"
    elif size_mm <= 10:
        risk = "modéré"
    else:
        risk = "élevé"

    # Le prompt doit être défini AVANT d'être utilisé
    prompt = f"""<|im_start|>system
Tu es un gastro-entérologue. Rédige un compte-rendu clinique court et professionnel en français.
<|im_end|>
<|im_start|>user
Données de l'IA de segmentation :
- Surface du polype : {polyp_area_ratio:.1%} de l'image.
- Taille estimée : {size_mm:.1f} mm.
- Niveau de risque associé : {risk}.

Rédige un compte-rendu avec exactement deux sections :
1. **Conclusion** (Analyse de la taille et du risque).
2. **Recommandations** (Que doit faire le médecin ?).

N'invente pas de détails sur l'image. Sois concis.
<|im_end|>
<|im_start|>assistant
"""
    
    # Appel du modèle avec le prompt
    output = llm_pipeline(prompt, do_sample=True, temperature=0.3, max_new_tokens=500)
    
    # Extraction de la réponse
    answer = output[0]['generated_text'].split("<|im_start|>assistant\n")[-1].strip()
    
    return answer