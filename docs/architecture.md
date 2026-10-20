# PolypCopilot — Schéma d'architecture

## Vue d'ensemble

```
                         ┌─────────────────────────────────────────────────┐
                         │                  UTILISATEUR                    │
                         │        Navigateur (image coloscopie)             │
                         └───────────────┬─────────────────────────────────┘
                                         │  upload image
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  KUBERNETES (Minikube / Kind)                                              │
│                                                                            │
│  ┌──────────────────────────── Service (ClusterIP :80) ─────────────────┐  │
│  │                    polypcopilot-api                                  │  │
│  └──────────────┬──────────────────────────────────────┬────────────────┘  │
│                 │ round-robin                          │                   │
│       ┌─────────▼──────────┐               ┌───────────▼─────────┐        │
│       │   Pod API (1/2)    │               │   Pod API (2/2)     │        │
│       │  ┌──────────────┐  │               │  ┌──────────────┐   │        │
│       │  │   FastAPI     │  │               │  │   FastAPI     │   │        │
│       │  │  /predict     │  │               │  │  /predict     │   │        │
│       │  │  /health      │  │               │  │  /health      │   │        │
│       │  │  /metrics     │  │               │  │  /metrics     │   │        │
│       │  │  /  (React)   │  │               │  │  /  (React)   │   │        │
│       │  └──────┬───────┘  │               │  └──────┬───────┘   │        │
│       │         ▼          │               │         ▼           │        │
│       │  ┌──────────────┐  │               │  ┌──────────────┐   │        │
│       │  │ U-Net + torch │  │               │  │ U-Net + torch │   │        │
│       │  │  (CPU, CPU)   │  │               │  │  (CPU, CPU)   │   │        │
│       │  └──────────────┘  │               │  └──────────────┘   │        │
│       └────────────────────┘               └─────────────────────┘        │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Prometheus (scrape /metrics toutes les 30s)  +  Grafana (dashboards)│  │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │ alertes                                  │
│                                  ▼                                          │
│                          Alertmanager (Slack / email)                       │
└────────────────────────────────────────────────────────────────────────────┘

                         CI/CD/CT (GitHub Actions)
                         ┌─────────────────────────────────────────────────┐
                         │  ci.yml : push → pytest (29 tests)             │
                         │  ct.yml : data/new/ → train → non-regression   │
                         │           → build Docker → push image          │
                         │           → kubectl apply → rolling update     │
                         └─────────────────────────────────────────────────┘
```

## Flux de données — Inférence

1. L'utilisateur sélectionne une image dans le **React** (port 5173 en dev, ou `/` derrière l'API en prod).
2. Le front envoie `POST /predict` (multipart/form-data) via `VITE_API_URL`.
3. Le **Service** K8s route vers un des 2 **pods** (round-robin).
4. Le pod décode l'image, applique `get_val_transforms()`, fait un forward U-Net.
5. Le masque binaire est redimensionné à la taille originale (nearest) et encodé en **base64**.
6. Le JSON retourne `{ mask_base64, polyp_area_ratio, inference_ms, model_version }`.
7. Le React dessine image + masque sur un **canvas** avec **curseur d'opacité**.

## Flux de données — Continuous Training

1. Un ingénieur copie de nouvelles images dans `data/new/` et push.
2. Le trigger `paths: data/new/**` lance **ct.yml**.
3. Le job CT : télécharge Kvasir-SEG → merge `data/new/` → ré-entraîne.
4. `evaluate.py` évalue le candidat (32 images test, seed 42).
5. **Gate** : `pytest tests/test_non_regression.py` vérifie `Dice_v2 ≥ Dice_v1 − 0.01`.
6. Si OK → upload artifact → build Docker → push image → `kubectl apply` → rolling update.
7. Si `promote_baseline=true` → la baseline est mise à jour dans Git.

## Monitoring & Alerting

| Métrique Prometheus | Type | Alerte |
|---|---|---|
| `polypcopilot_model_val_dice` | Gauge | **Dice < 0.85** (critical, for: 5m) |
| `polypcopilot_predictions_total` | Counter | Taux d'erreur > 5% (warning, for: 5m) |
| `polypcopilot_inference_seconds_bucket` | Histogram | P95 > 1s (warning, for: 10m) |
| `polypcopilot_polyp_area_ratio` | Gauge | Surface > 95% (warning, drift) |
| `up{job="polypcopilot-api"}` | Standard | API down (critical, for: 1m) |

## Décisions d'architecture

| Décision | Justification |
|---|---|
| Front servi par l'API (`StaticFiles`) | Une seule image, un seul Service, pas de CORS en prod |
| `readinessProbe` sur `/health` | Le pod n'accepte le trafic qu'une fois le modèle chargé |
| `livenessProbe` sur `/health` | Redémarre un pod gelé |
| CPU-only dans le Docker | Pas de GPU sur Minikube/Kind, image 3× plus légère qu'avec CUDA |
| Baseline versionnée dans Git | Auditable, historique des modèles, rollback possible |
| `paths-ignore: pipelines/baseline.json` dans CI | Évite la boucle infinie CI↔CT |
