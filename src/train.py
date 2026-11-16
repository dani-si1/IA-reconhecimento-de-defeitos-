"""
Treino do detector de defeitos em PCB (DeepPCB) com YOLO26n na CPU.

Os valores padrão reproduzem exatamente o treino documentado no README
(results/yolo26n/args.yaml). Rode a partir da raiz do repositório:

    python src/train.py
    python src/train.py --patience 0          # desliga a parada antecipada
    python src/train.py --device 0 --batch 32 # se tiver GPU NVIDIA
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description="Treina o YOLO26 no DeepPCB")
    ap.add_argument("--model", default="yolo26n.pt", help="pesos pré-treinados (COCO) de partida")
    ap.add_argument("--data", default=str(ROOT / "data" / "deeppcb.yaml"))
    ap.add_argument("--imgsz", type=int, default=832)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="yolo26n_cpu_max")
    return ap.parse_args()


def main():
    a = parse_args()
    model = YOLO(a.model)
    model.train(
        data=a.data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        patience=a.patience,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        cache="ram",
        # otimização
        optimizer="auto",   # com ~9,5 mil iterações a Ultralytics escolheu AdamW(lr=0.001)
        cos_lr=True,
        close_mosaic=30,    # desliga o mosaic nas últimas 30 épocas
        # aumento de dados pensado para imagens binarizadas de PCB:
        # sem variação de cor/rotação livre (as trilhas são ortogonais/45°),
        # mas com espelhamentos, pequena translação e escala
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
        degrees=0.0, shear=0.0, perspective=0.0,
        translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        project=str(ROOT / "runs"),
        name=a.name,
        plots=True,
    )


if __name__ == "__main__":  # necessário no Windows por causa dos workers do DataLoader
    main()
