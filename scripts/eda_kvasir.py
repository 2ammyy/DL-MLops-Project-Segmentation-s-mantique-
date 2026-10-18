import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "raw" / "Kvasir-SEG"


def report():
    print("=" * 62)
    print("RAPPORT EDA - KVASIR-SEG")
    print("=" * 62)
    print(f"Source : https://datasets.simula.no/kvasir-seg/")
    print("Cite   : Jha et al., Kvasir-SEG, MMM 2020, DOI 10.1007/978-3-030-37734-2_37")
    print("Licence: recherche/education uniquement, citation obligatoire")
    print()

    images = sorted((DATA / "images").glob("*.jpg"))
    masks = sorted((DATA / "masks").glob("*.jpg"))
    stems_img = {p.stem for p in images}
    stems_msk = {p.stem for p in masks}
    print("-- Contenu --")
    print(f"images      : {len(images)} fichiers JPEG")
    print(f"masks       : {len(masks)} fichiers JPEG")
    print(f"paires      : {len(stems_img & stems_msk)} (orphelines: {len(stems_img ^ stems_msk)})")
    total_mb = sum(p.stat().st_size for p in images + masks) / 1e6
    print(f"poids total : {total_mb:.1f} Mo")

    json_path = DATA / "kavsir_bboxes.json"
    if json_path.exists():
        boxes = json.loads(json_path.read_text())
        print()
        print("-- kavsir_bboxes.json (coquille officielle 'kavsir') --")
        first_key = next(iter(boxes))
        print(f"entrees     : {len(boxes)}")
        print(f"variables   : {sorted(boxes[first_key])}")
        print(f"exemple     : {first_key} -> {boxes[first_key]}")

    print()
    print("-- Analyse pixel --")
    shapes, areas, mid_vals = [], [], 0
    for p in images:
        im = cv2.imread(str(p))
        shapes.append(im.shape[:2])
        m = cv2.imread(str(DATA / "masks" / f"{p.stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        areas.append((m > 127).mean())
        vals = np.unique(m)
        mid_vals += int(((vals > 0) & (vals < 255)).sum() > 0)

    shapes = np.array(shapes)
    areas = np.array(areas)
    print(f"resolution  : {shapes.min(0)[1]}x{shapes.min(0)[0]} a {shapes.max(0)[1]}x{shapes.max(0)[0]}")
    print(f"surface polype: moy={areas.mean()*100:.1f}% | med={np.median(areas)*100:.1f}%"
          f" | min={areas.min()*100:.1f}% | max={areas.max()*100:.1f}%")
    print(f"images < 5% : {(areas < 0.05).sum()} (fond dominant)")
    print(f"masques avec valeurs intermediaires (artefacts JPEG): {mid_vals}/1000")
    print("  -> d'ou le seuil (mask > 127) dans src/data/dataset.py")

    from src.data.dataset import get_datasets

    train_ds, val_ds, test_ds = get_datasets(DATA / "images", DATA / "masks", seed=42)
    print()
    print("-- Decoupage (seed=42, 70/10/20) --")
    print(f"train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print("=" * 62)


def montage(n=6, tile=256, out="data/preview_kvasir.jpg"):
    images = sorted((DATA / "images").glob("*.jpg"))
    idx = np.linspace(0, len(images) - 1, n).astype(int)
    rows = []
    for i in idx:
        p = images[i]
        im = cv2.resize(cv2.imread(str(p)), (tile, tile))
        m = cv2.imread(str(DATA / "masks" / f"{p.stem}.jpg"), 0)
        m = cv2.resize(m, (tile, tile), interpolation=cv2.INTER_NEAREST)
        overlay = im.copy()
        sel = m > 127
        overlay[sel] = (0.4 * overlay[sel] + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        mask_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        rows.append(np.hstack([im, mask_bgr, overlay]))
    canvas = np.vstack(rows)
    cv2.putText(canvas, "IMAGE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "MASQUE", (tile + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "OVERLAY", (2 * tile + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out_path = ROOT / out
    cv2.imwrite(str(out_path), canvas)
    print(f"Montage {n} paires -> {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    report()
    montage()
