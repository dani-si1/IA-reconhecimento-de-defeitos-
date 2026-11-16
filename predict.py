"""
Roda o detector treinado em uma imagem ou pasta de imagens.

    python src/predict.py caminho/para/imagem.jpg
    python src/predict.py caminho/para/pasta --conf 0.5
    python src/predict.py foto_da_placa.jpg --binarize        # aproxima a foto do estilo DeepPCB

Saídas em outputs/: a imagem anotada (<nome>_pred.png) e, com --binarize,
a entrada binarizada que foi de fato enviada ao modelo (<nome>_bin.png).

Sobre --binarize: o DeepPCB é composto de imagens binárias (cobre preto, substrato
branco) obtidas com scanner linear após alinhamento e limiarização. Uma foto comum
precisa passar por algo parecido para ficar perto da distribuição de treino.
Use --no-invert se, na sua foto, o cobre já aparecer mais escuro que o substrato.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def binarize(img_bgr, blur=9, invert=True):
    """Tons de cinza -> mediana (remove pontinhos) -> limiar de Otsu -> 3 canais."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur and blur > 1:
        gray = cv2.medianBlur(gray, blur if blur % 2 else blur + 1)  # precisa ser ímpar
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)  # a YOLO espera 3 canais


def main():
    ap = argparse.ArgumentParser(description="Detecção de defeitos em PCB")
    ap.add_argument("source", type=Path, help="imagem ou pasta")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="limiar de confiança (0.5 maximiza o F1 na validação)")
    ap.add_argument("--imgsz", type=int, default=832, help="mesmo tamanho usado no treino")
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--blur", type=int, default=9, help="janela da mediana (ímpar)")
    ap.add_argument("--no-invert", dest="invert", action="store_false")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in EXTS) \
        if args.source.is_dir() else [args.source]
    if not files:
        raise SystemExit(f"Nenhuma imagem encontrada em {args.source}")
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[aviso] não foi possível abrir {path}")
            continue
        entrada = binarize(img, args.blur, args.invert) if args.binarize else img
        if args.binarize:
            cv2.imwrite(str(args.out / f"{path.stem}_bin.png"), entrada)

        r = model.predict(entrada, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        print(f"\n{path.name}: {len(r.boxes)} detecções")
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  {model.names[int(box.cls)]:<10} {float(box.conf):.2f}  ({x1}, {y1}, {x2}, {y2})")
        cv2.imwrite(str(args.out / f"{path.stem}_pred.png"), r.plot())

    print(f"\nImagens anotadas salvas em {args.out}")


if __name__ == "__main__":
    main()
