import os
import shutil
import random
from pathlib import Path

# Configuration des chemins
RAW_DIR = Path("data/raw/Kvasir-SEG")
PROCESSED_DIR = Path("data/processed")

# Ratios (70% train, 10% val, 20% test)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

def split_data():
    print("Début du split des données...")
    
    # 1. Vérifier que les données brutes existent
    images_dir = RAW_DIR / "images"
    masks_dir = RAW_DIR / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"Erreur : Les dossiers {images_dir} ou {masks_dir} n'existent pas.")
        return

    # 2. Récupérer tous les noms de fichiers (sans extension)
    images = sorted([f.stem for f in images_dir.glob("*.jpg")])
    
    if len(images) == 0:
        print("Erreur : Aucune image trouvée.")
        return
        
    random.seed(42) # Pour la reproductibilité
    random.shuffle(images)
    
    # 3. Calculer les tailles
    total = len(images)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    
    train_files = images[:train_size]
    val_files = images[train_size:train_size + val_size]
    test_files = images[train_size + val_size:]
    
    # 4. Créer les dossiers de destination
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    for split_name, files in splits.items():
        split_img_dir = PROCESSED_DIR / split_name / "images"
        split_mask_dir = PROCESSED_DIR / split_name / "masks"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        for name in files:
            # Copier l'image
            shutil.copy2(images_dir / f"{name}.jpg", split_img_dir / f"{name}.jpg")
            # Copier le masque
            shutil.copy2(masks_dir / f"{name}.jpg", split_mask_dir / f"{name}.jpg")
            
        print(f"  -> {split_name} : {len(files)} images")

    print("Split terminé avec succès !")

if __name__ == "__main__":
    split_data()