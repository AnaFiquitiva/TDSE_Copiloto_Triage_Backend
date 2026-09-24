# Resultados del diseño experimental: 2x2 (E1-E4), ruta principal (E5-E6) y línea base (C0)

Generado automáticamente por `experiments/run_experiment.py`. Estas cifras son el resultado de ejecutar el pipeline sobre las 74 viñetas sintéticas de `data/cases.json` (dentro del rango de 60-100 casos del protocolo del documento del proyecto, pero con doble ciego intra-equipo, no evaluadores clínicos externos); no constituyen una validación clínica externa (ver `README.md`, sección de limitaciones).

Convenciones: S y sobre-triage se calculan solo sobre los casos en que la celda responde; la sensibilidad I-II cuenta la abstención como fallo. Los intervalos son IC bootstrap del 95 % (2000 réplicas por casos, semilla 42). 'S par. (celda / C0)' compara S sobre el subconjunto en que ambos responden.

## Acuerdo intra-equipo (gold standard)

- Pares evaluados: 68
- Kappa ponderado cuadrático (κ_w): 0.953

## Línea base C0 (árbol de reglas, sin RAG)

- Casos elegibles: 66
- S (sub-triage ponderado): 0.394 [0.242–0.561]
- Sensibilidad niveles I-II: 0.455 [0.273–0.606]
- Sobre-triage: 0.273
- Acuerdo exacto: 0.530
- Fidelidad de citación: 0.227
- S por nivel de referencia: {1: 0.8888888888888888, 2: 0.6666666666666666, 3: 0.0, 4: 0.0, 5: 0.0}

## Celdas del copiloto

| Celda | Corpus | Recuperación | N eleg. | Cobertura | S [IC95] | S par. (celda / C0, n) | Sensib. I-II [IC95] | McNemar vs C0 (p) | Recall@1 | Recall@k [IC95] | MRR | Fidelidad | Abst. correcta | Abst. indebida | Sobre-triage | Completitud | Degradado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | raw | generic | 66 | 0.47 | 0.258 [0.097–0.419] | 0.258 / 0.516 (n=31) | 0.485 [0.303–0.667] | 1.000 (6:5) | 0.258 | 0.500 [0.379–0.621] | 0.353 | 0.387 | 0.500 | 0.530 | 0.258 | 1.00 | 0.00 |
| E2 | reformatted | generic | 66 | 0.41 | 0.074 [0.000–0.185] | 0.074 / 0.333 (n=27) | 0.212 [0.091–0.364] | 0.077 (4:12) | 0.288 | 0.561 [0.439–0.682] | 0.387 | 0.407 | 0.500 | 0.591 | 0.519 | 1.00 | 0.00 |
| E3 | raw | clinical_es | 66 | 0.50 | 0.212 [0.091–0.364] | 0.212 / 0.545 (n=33) | 0.485 [0.303–0.667] | 1.000 (8:7) | 0.379 | 0.621 [0.500–0.742] | 0.464 | 0.545 | 0.500 | 0.500 | 0.212 | 1.00 | 0.00 |
| E4 | reformatted | clinical_es | 66 | 0.56 | 0.108 [0.000–0.243] | 0.108 / 0.351 (n=37) | 0.394 [0.242–0.545] | 0.774 (5:7) | 0.409 | 0.667 [0.561–0.773] | 0.497 | 0.541 | 0.500 | 0.439 | 0.459 | 1.00 | 0.00 |
| E5 | reformatted | rrf(bm25+gemini) + LLM | 66 | 0.85 | 0.000 [0.000–0.000] | 0.000 / 0.375 (n=56) | 0.879 [0.758–0.970] | 0.000 (14:0) | 0.561 | 0.833 [0.742–0.924] | 0.656 | 0.929 | 1.000 | 0.152 | 0.018 | 1.00 | 0.00 |
| E6 | reformatted | rrf(bm25+gemini) + LLM + guardarraíl | 66 | 0.85 | 0.000 [0.000–0.000] | 0.000 / 0.375 (n=56) | 0.879 [0.758–0.970] | 0.000 (14:0) | 0.561 | 0.833 [0.742–0.924] | 0.656 | 0.929 | 1.000 | 0.152 | 0.018 | 1.00 | 0.00 |

McNemar vs C0 (H1): valor p exacto sobre los casos de referencia I-II; entre paréntesis, casos que solo la celda clasifica en I-II : casos que solo C0 clasifica en I-II.
E1-E4: ruta determinista explícita, sin guardarraíl (estudio de recuperación, sin red). E5-E6: ruta principal (BM25 + Gemini fusionados por RRF, y LLM que elige la cita), sin y con guardarraíl; sus métricas se calculan solo sobre los casos que respondió Gemini ('Degradado' = fracción que cayó al respaldo).

### Repeticiones de E5 y E6 (media [mín–máx])

3 repeticiones; modos de caché: readonly, readonly, readonly.

| Celda | cobertura | S_sub_triage | sensibilidad_I_II | recall_at_1 | recall_at_k | mrr | sobre_triage | tasa_modo_degradado |
|---|---|---|---|---|---|---|---|---|
| E5 | 0.848 [0.848–0.848] | 0.000 [0.000–0.000] | 0.879 [0.879–0.879] | 0.561 [0.561–0.561] | 0.833 [0.833–0.833] | 0.656 [0.656–0.656] | 0.018 [0.018–0.018] | 0.000 [0.000–0.000] |
| E6 | 0.848 [0.848–0.848] | 0.000 [0.000–0.000] | 0.879 [0.879–0.879] | 0.561 [0.561–0.561] | 0.833 [0.833–0.833] | 0.656 [0.656–0.656] | 0.018 [0.018–0.018] | 0.000 [0.000–0.000] |

### H5: efecto del guardarraíl (E6 frente a E5, pareado)

- Casos elegibles pareados: 66 (ambas responden: 56)
- S pareado: E6 0.000 / E5 0.000
- Sobre-triage pareado: E6 0.018 / E5 0.018
- Sensibilidad I-II, McNemar exacto: p = 1.000 (solo E6 acierta: 0; solo E5 acierta: 0)

### Versiones

- Modelo de generación: `gemini-3.5-flash-lite`; embeddings: `gemini-embedding-001`
- Prompt: `triage-selector-v1`; reglas de alarma: `alarmas-v1`

### S desglosado por nivel de referencia, por celda

- **E1**: {1: 0.08333333333333333, 2: 0.6, 3: 0.0, 4: 0.5, 5: 0.0}
- **E2**: {1: 0.0, 2: 0.6666666666666666, 3: 0.0, 4: 0.0, 5: 0.0}
- **E3**: {1: 0.07692307692307693, 2: 0.625, 3: 0.0, 4: 0.2, 5: 0.0}
- **E4**: {1: 0.2, 2: 0.3333333333333333, 3: 0.0, 4: 0.0, 5: 0.0}
- **E5**: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
- **E6**: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
