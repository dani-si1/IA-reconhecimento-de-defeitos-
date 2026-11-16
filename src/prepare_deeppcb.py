"""
Converte o DeepPCB original (https://github.com/tangsanli5201/DeepPCB) para o formato YOLO.

Estrutura esperada (pasta PCBData do repositório original):
    PCBData/
      trainval.txt   ->  linhas "groupXXXXX/XXXXX/XXXXXXXX.jpg groupXXXXX/XXXXX_not/XXXXXXXX.txt"
      test.txt
      group00041/00041/00041000_test.jpg   (imagem com defeitos, a que usamos)
      group00041/00041/00041000_temp.jpg   (template sem defeito, não usado aqui)
      group00041/00041_not/00041000.txt    (anotações "x1,y1,x2,y2,tipo", tipo de 1 a 6)

Saída (formato Ultralytics), dentro de data/:
    data/images/train/*.jpg   data/labels/train/*.txt   <- trainval.txt (1000 imagens)
    data/images/val/*.jpg     data/labels/val/*.txt     <- test.txt     (500 imagens)

Uso:
    python src/prepare_deeppcb.py --src caminho/para/DeepPCB/PCBData
"""
import argparse
import re
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]  # tipos 1..6 do DeepPCB


def find_image(src: Path, rel: str) -> Path:
    p = src / rel
    candidates = [p, p.with_name(p.stem + "_test" + p.suffix)]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"imagem não encontrada: {candidates}")


def convert_split(src: Path, list_file: str, split: str, out: Path) -> tuple[int, int]:
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for line in (src / list_file).read_text().split("\n"):
        if not line.strip():
            continue
        img_rel, ann_rel = line.split()[:2]
        img_path = find_image(src, img_rel)
        h, w = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).shape[:2]
        stem = Path(img_rel).stem.replace("_test", "")

        rows = []
        for ann in (src / ann_rel).read_text().split("\n"):
            vals = [v for v in re.split(r"[,\s]+", ann.strip()) if v]
            if len(vals) < 5:
                continue
            x1, y1, x2, y2, t = map(int, map(float, vals[:5]))
            if not 1 <= t <= 6:
                continue
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
            rows.append(f"{t - 1} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        shutil.copy2(img_path, img_dir / f"{stem}{img_path.suffix}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n_img += 1
        n_box += len(rows)
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="pasta PCBData do DeepPCB")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    a = ap.parse_args()
    for list_file, split in (("trainval.txt", "train"), ("test.txt", "val")):
        n_img, n_box = convert_split(a.src, list_file, split, a.out)
        print(f"{split}: {n_img} imagens, {n_box} defeitos anotados")


if __name__ == "__main__":
    main()
