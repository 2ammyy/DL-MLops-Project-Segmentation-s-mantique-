from prometheus_client import Counter, Gauge, Histogram

PREDICTION_COUNT = Counter(
    "polypcopilot_predictions_total",
    "Nombre total de predictions effectuees",
    ["status"],
)
INFERENCE_SECONDS = Histogram(
    "polypcopilot_inference_seconds",
    "Duree d'inference en secondes",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
POLYP_AREA_GAUGE = Gauge(
    "polypcopilot_polyp_area_ratio",
    "Derniere surface de polype predite (ratio 0-1)",
)
MODEL_DICE_GAUGE = Gauge(
    "polypcopilot_model_val_dice",
    "Dice score de validation du modele charge",
)
