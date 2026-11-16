"""
Gera os gráficos de análise do treino a partir do results.csv da Ultralytics.

Uso:
    python src/analyze_results.py
    python src/analyze_results.py --csv results/yolo26n/results.csv --out results/analise

Saídas (em --out):
    curva_map.png             mAP50 x mAP50-95 por época, com média móvel e marcos do treino
    box_loss_vs_map.png       dispersão val/box_loss x mAP50-95 (mostra que a oscilação é de localização)
    metricas_por_classe.png   mAP50 e mAP50-95 de cada classe (validação final do best.pt)
    resumo.md                 números usados no README
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Paleta e estilo (tons validados para daltonismo; texto sempre em tinta neutra)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"    # mAP50-95
ORANGE = "#eb6834"  # mAP50

# Métricas por classe da validação final do best.pt (copiadas do train_log.txt)
PER_CLASS = pd.DataFrame(
    [
        ("open", 659, 0.968, 0.941, 0.981, 0.644),
        ("short", 478, 0.817, 0.866, 0.904, 0.583),
        ("mousebite", 586, 0.977, 0.944, 0.972, 0.680),
        ("spur", 483, 0.970, 0.892, 0.971, 0.665),
        ("copper", 464, 0.984, 0.946, 0.987, 0.819),
        ("pin-hole", 470, 0.910, 0.966, 0.985, 0.789),
    ],
    columns=["classe", "instancias", "P", "R", "mAP50", "mAP50-95"],
)


def setup_style():
    family = "DejaVu Sans"
    if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
        family = "Inter"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK,
        }
    )


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["fitness"] = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    df["map5095_mm10"] = df["metrics/mAP50-95(B)"].rolling(10, min_periods=1).mean()
    return df


def plot_curva_map(df: pd.DataFrame, out: Path, epochs_planned=150, close_mosaic=30, patience=80):
    best = df.loc[df["fitness"].idxmax()]
    last = df.iloc[-1]
    mosaic_off = epochs_planned - close_mosaic + 1

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
    ax.axvspan(last["epoch"] + 0.5, epochs_planned, color="#f0efec", zorder=0, lw=0)
    ax.text(
        (last["epoch"] + mosaic_off) / 2, 0.16,
        f"épocas {int(last['epoch']) + 1}–{epochs_planned}\nnão executadas",
        ha="center", va="center", color=INK_2, fontsize=10,
    )
    ax.axvline(mosaic_off, color=INK_2, lw=1, ls=":")
    ax.text(mosaic_off + 1, 0.30, f"close_mosaic\n(desligaria o mosaic\nna época {mosaic_off})",
            color=INK_2, fontsize=9, va="bottom")

    ax.plot(df["epoch"], df["metrics/mAP50(B)"], color=ORANGE, lw=2, label="mAP50")
    ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], color=BLUE, lw=1, alpha=0.45,
            label="mAP50-95 (por época)")
    ax.plot(df["epoch"], df["map5095_mm10"], color=BLUE, lw=2.6,
            label="mAP50-95 (média móvel de 10 épocas)")

    # marcos
    ax.scatter([best["epoch"]], [best["metrics/mAP50-95(B)"]], s=70, color=BLUE,
               edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.annotate(
        f"best.pt — época {int(best['epoch'])}\nmAP50-95 = {best['metrics/mAP50-95(B)']:.3f}",
        xy=(best["epoch"], best["metrics/mAP50-95(B)"]), xytext=(22, 0.80),
        color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK_2, lw=1),
    )
    ax.axvline(last["epoch"], color=INK_2, lw=1, ls="--")
    ax.text(last["epoch"] - 1, 0.04,
            f"parada antecipada na época {int(last['epoch'])}\n(patience={patience} sem superar a época {int(best['epoch'])})",
            ha="right", va="bottom", color=INK_2, fontsize=9)

    ax.set_xlim(0, epochs_planned)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("época")
    ax.set_ylabel("métrica na validação (500 imagens)")
    ax.set_title("O mAP50 estabiliza cedo; o mAP50-95 oscila e ainda subia quando o treino parou")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "curva_map.png")
    plt.close(fig)


def plot_box_loss(df: pd.DataFrame, out: Path):
    x, y = df["val/box_loss"], df["metrics/mAP50-95(B)"]
    r = x.corr(y)
    late = df[df["epoch"] >= 6]
    r_late = late["val/box_loss"].corr(late["metrics/mAP50-95(B)"])
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    ax.scatter(x, y, s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.2, alpha=0.9)
    marks = {
        int(df.loc[y.idxmax(), "epoch"]): "época {ep} (best.pt)",
        int(df.loc[y.idxmin(), "epoch"]): "época {ep}",
        1: "época 1 (aquecimento)",
    }
    for ep, txt in marks.items():
        row = df[df["epoch"] == ep].iloc[0]
        ax.annotate(txt.format(ep=ep), xy=(row["val/box_loss"], row["metrics/mAP50-95(B)"]),
                    xytext=(8, 4), textcoords="offset points", color=INK, fontsize=9)
    ax.text(0.98, 0.95, f"Pearson r = {r:.2f} (todas as épocas)\nr = {r_late:.2f} (a partir da época 6)",
            transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=10.5, linespacing=1.5)
    ax.set_xlabel("val/box_loss (erro de regressão da caixa)")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Épocas com mAP50-95 baixo são épocas com\nerro de caixa alto na validação")
    fig.tight_layout()
    fig.savefig(out / "box_loss_vs_map.png")
    plt.close(fig)
    return r


def plot_per_class(out: Path):
    d = PER_CLASS.sort_values("mAP50-95", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    h = 0.36
    ypos = range(len(d))
    ax.barh([i + h / 2 + 0.01 for i in ypos], d["mAP50"], height=h, color=ORANGE, label="mAP50")
    ax.barh([i - h / 2 - 0.01 for i in ypos], d["mAP50-95"], height=h, color=BLUE, label="mAP50-95")
    for i, v in zip(ypos, d["mAP50-95"]):
        ax.text(v + 0.01, i - h / 2 - 0.01, f"{v:.3f}", va="center", color=INK, fontsize=9)
    for i, v in zip(ypos, d["mAP50"]):
        ax.text(v + 0.01, i + h / 2 + 0.01, f"{v:.3f}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(d["classe"], color=INK)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("métrica na validação (best.pt)")
    ax.set_title("Todas as classes passam de 0.90 em mAP50;\n'short' é a mais difícil")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "metricas_por_classe.png")
    plt.close(fig)


def write_summary(df: pd.DataFrame, r: float, out: Path):
    best = df.loc[df["fitness"].idxmax()]
    last = df.iloc[-1]
    after = df[df["epoch"] > best["epoch"]]
    cols = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    lines = [
        "# Resumo numérico (gerado por src/analyze_results.py)",
        "",
        f"- épocas executadas: {int(last['epoch'])}",
        f"- tempo total: {last['time'] / 3600:.2f} h  |  tempo mediano por época: {df['time'].diff().median():.0f} s",
        f"- melhor época (fitness = 0,1·mAP50 + 0,9·mAP50-95): {int(best['epoch'])}",
        f"- mAP50 máximo: {df['metrics/mAP50(B)'].max():.3f} (época {int(df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch'])})",
        f"- após a época {int(best['epoch'])}: mAP50-95 entre {after['metrics/mAP50-95(B)'].min():.3f} e {after['metrics/mAP50-95(B)'].max():.3f}"
        f" (desvio-padrão {after['metrics/mAP50-95(B)'].std():.3f}); mAP50 com desvio-padrão {after['metrics/mAP50(B)'].std():.3f}",
        f"- média móvel (10 épocas) do mAP50-95: {df['map5095_mm10'].iloc[19]:.3f} (ép. 11–20) → "
        f"{df['map5095_mm10'].iloc[49]:.3f} (ép. 41–50) → {df['map5095_mm10'].iloc[-1]:.3f} (ép. {int(last['epoch']) - 9}–{int(last['epoch'])})",
        f"- correlação val/box_loss × mAP50-95: r = {r:.2f}",
        f"- correlação val/l1_loss × mAP50-95: r = {df['val/l1_loss'].corr(df['metrics/mAP50-95(B)']):.2f}",
        "- a partir da época 6: r(val/box_loss) = {:.2f}; r(val/cls_loss) = {:.2f}".format(
            df[df.epoch >= 6]["val/box_loss"].corr(df[df.epoch >= 6]["metrics/mAP50-95(B)"]),
            df[df.epoch >= 6]["val/cls_loss"].corr(df[df.epoch >= 6]["metrics/mAP50-95(B)"])),
        "",
        "| pesos | época | P | R | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|",
        "| best.pt | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(int(best["epoch"]), *best[cols]),
        "| last.pt | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(int(last["epoch"]), *last[cols]),
    ]
    (out / "resumo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=root / "results/yolo26n/results.csv")
    ap.add_argument("--out", type=Path, default=root / "results/analise")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    setup_style()
    df = load(args.csv)
    plot_curva_map(df, args.out)
    r = plot_box_loss(df, args.out)
    plot_per_class(args.out)
    write_summary(df, r, args.out)


if __name__ == "__main__":
    main()
