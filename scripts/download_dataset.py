import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
DATA_DIR = ROOT / "data" / "raw"
TARGET_DIR = DATA_DIR / "Kvasir-SEG"


def download(dest=None):
    import urllib.request

    dest = Path(dest) if dest else ROOT / "data" / "kvasir-seg.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Deja telecharge : {dest}")
        return dest
    print(f"Telechargement {SOURCE_URL} ...")
    urllib.request.urlretrieve(SOURCE_URL, dest)
    print(f"OK -> {dest} ({dest.stat().st_size / 1e6:.1f} Mo)")
    return dest


def extract(zip_path=None):
    zip_path = Path(zip_path) if zip_path else ROOT / "data" / "kvasir-seg.zip"
    if TARGET_DIR.exists():
        print(f"Deja extrait : {TARGET_DIR}")
        return TARGET_DIR
    print("Extraction ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    extracted = DATA_DIR / "Kvasir-SEG"
    if extracted.name != "Kvasir-SEG":
        raise SystemExit(f"Structure inattendue dans le zip : {extracted}")
    print(f"OK -> {extracted}")
    return extracted


def verify():
    n_img = len(list((TARGET_DIR / "images").glob("*.jpg")))
    n_msk = len(list((TARGET_DIR / "masks").glob("*.jpg")))
    if n_img != 1000 or n_msk != 1000:
        raise SystemExit(f"Dataset incomplet : {n_img} images / {n_msk} masques")
    print(f"Verification OK : {n_img} images / {n_msk} masques")


if __name__ == "__main__":
    if not TARGET_DIR.exists():
        zip_path = download()
        extract(zip_path)
    verify()
