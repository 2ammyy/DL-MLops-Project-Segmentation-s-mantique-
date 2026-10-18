# PolypCopilot 🏥

**Segmentation sémantique de polypes colorectaux** — Projet fusionné Deep Learning 2 + MLOps.

## 🎯 Objectif

Entraîner un modèle de segmentation (U-Net & variantes via `segmentation-models-pytorch`) sur le dataset [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/), puis industrialiser toute la chaîne : API FastAPI → Frontend React → CI/CD/CT → Kubernetes.

## 🏗️ Architecture

```
Image coloscopie → FastAPI /predict → Modèle U-Net → Masque → React (curseur d'opacité)
                          ↑
        CI/CD/CT (GitHub Actions) → Docker → Kubernetes (2 replicas)
                          ↑
              data/new/ (nouvelles données → retraining)
```

## 📂 Structure

```
├── src/            # Code ML (data, modèles, loss, metrics, train)
├── api/            # Service FastAPI d'inférence
├── frontend/       # Interface React (Vite) + curseur d'opacité
├── k8s/            # Manifestes Kubernetes (Deployment 2 replicas + Service)
├── tests/          # Tests unitaires + non-régression
├── pipelines/      # Orchestration CI/CD/CT
├── monitoring/     # Règles d'alerting (Prometheus)
├── data/           # Dataset Kvasir-SEG (non versionné)
├── models/         # Checkpoints & registre (non versionné)
└── .github/        # Workflows GitHub Actions
```

## 🚀 Démarrage rapide

```bash
# 1. Dépendances
pip install -r requirements.txt

# 2. Entraînement
python -m src.train --epochs 20 --model unet

# 3. API
uvicorn api.main:app --reload

# 4. Frontend
cd frontend && npm install && npm run dev
```

## 📊 Métriques

| Métrique | Cible |
|----------|-------|
| Dice Score | ≥ 0.90 |
| IoU | ≥ 0.85 |
| Non-régression | Dice_v2 ≥ Dice_v1 − 0.01 |
| Alerte | Dice < 0.85 |

## 📖 Rapport

Voir [CHANGELOG.md](CHANGELOG.md) pour l'historique des versions.
