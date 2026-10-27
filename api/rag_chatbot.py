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