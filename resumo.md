# Resumo numérico (gerado por src/analyze_results.py)

- épocas executadas: 91
- tempo total: 6.01 h  |  tempo mediano por época: 239 s
- melhor época (fitness = 0,1·mAP50 + 0,9·mAP50-95): 11
- mAP50 máximo: 0.977 (época 67)
- após a época 11: mAP50-95 entre 0.173 e 0.669 (desvio-padrão 0.093); mAP50 com desvio-padrão 0.026
- média móvel (10 épocas) do mAP50-95: 0.460 (ép. 11–20) → 0.500 (ép. 41–50) → 0.589 (ép. 82–91)
- correlação val/box_loss × mAP50-95: r = -0.89
- correlação val/l1_loss × mAP50-95: r = -0.90
- a partir da época 6: r(val/box_loss) = -0.98; r(val/cls_loss) = -0.91

| pesos | época | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| best.pt | 11 | 0.937 | 0.928 | 0.967 | 0.697 |
| last.pt | 91 | 0.969 | 0.953 | 0.977 | 0.639 |
