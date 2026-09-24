# Comparación de rutas: determinista vs. Gemini

Generado por `experiments/compare_backends.py` sobre las 74 viñetas de `data/cases.json`, con corpus `reformatted`.

Ruta realmente usada al pedir Gemini, por caso: {'deterministic_fallback': 74}. Los casos que cayeron al respaldo (`deterministic_fallback`) se reportan en su propia columna y NO se mezclan con los de Gemini.

| Métrica | Gemini (casos que respondió) | Determinista (mismos casos) | Determinista (todos) | Respaldo (casos degradados) |
|---|---|---|---|---|
| Casos | — | — | 74 | 74 |
| Casos elegibles | — | — | 66 | 66 |
| Cobertura | — | — | 0.56 | 0.56 |
| S (sub-triage ponderado) | — | — | 0.027 | 0.027 |
| Sensibilidad I-II | — | — | 0.455 | 0.455 |
| Recall@1 | — | — | 0.409 | 0.409 |
| Recall@k | — | — | 0.667 | 0.667 |
| MRR | — | — | 0.497 | 0.497 |
| Sobre-triage | — | — | 0.459 | 0.459 |
| Abstención correcta | — | — | 0.500 | 0.500 |
| Abstención indebida | — | — | 0.439 | 0.439 |

**Gemini no respondió ningún caso** (sin `GEMINI_API_KEY` ni caché de respuestas): no hay comparación que reportar.

Con 74 casos esta comparación es ilustrativa, no concluyente estadísticamente.
