// Tipos compartidos, en espejo de los modelos Pydantic de web/backend/app.py.
// Mantenerlos sincronizados manualmente es aceptable dado el tamaño del
// proyecto; si creciera, valdría la pena generarlos desde el esquema OpenAPI.

export type CorpusFormat = 'raw' | 'reformatted'
export type EmbeddingMode = 'generic' | 'clinical_es'
export type Backend = 'deterministic' | 'gemini'

export interface Suggestion {
  level: number | null
  citation: string | null
  confidence: number
  abstained: boolean
  reason: string
}

export interface RetrievedFragment {
  chunk_id: string
  score: number
  text: string
  level: number | null
}

export interface BaselineResult {
  level: number
  citation: string
  matched_keyword: string | null
}

export interface ConsultaResponse {
  suggestion: Suggestion
  retrieved: RetrievedFragment[]
  baseline: BaselineResult
  backend_used: string
  disclaimer: string
}

// Las métricas pueden venir en null (NaN en Python: p. ej. sin casos I-II).
// Una celda E5/E6 sin API ni caché trae solo `no_evaluada` con el motivo.
export interface CellResult {
  cell: string
  corpus_format: string
  embedding_mode: string
  no_evaluada?: string
  n_eligible_cases?: number
  n_abstained_among_eligible?: number
  cobertura?: number | null
  S_sub_triage?: number | null
  S_por_nivel?: Record<string, number>
  sensibilidad_I_II?: number | null
  recall_at_1?: number | null
  recall_at_k?: number | null
  mrr?: number | null
  fidelidad_citacion?: number | null
  tasa_abstencion_correcta?: number | null
  tasa_abstencion_indebida?: number | null
  sobre_triage?: number | null
  tasa_modo_degradado?: number | null
}

export interface BaselineCellResult {
  cell: string
  descripcion: string
  n_eligible_cases: number
  S_sub_triage: number
  S_por_nivel: Record<string, number>
  sensibilidad_I_II: number
  fidelidad_citacion: number
  sobre_triage?: number | null
}

export interface KappaResult {
  n_pares: number
  kappa_ponderado_cuadratico: number
}

export interface ExperimentoResponse {
  kappa: KappaResult
  baseline: BaselineCellResult
  cells: CellResult[]
  n_total_cases: number
}

export interface DurationStat {
  n: number
  median: number
  mean: number
}

export interface Chi2Test {
  chi2: number
  df: number
  p_value: number
}

export interface DatasetResponse {
  total: number
  distribution: Record<string, number>
  duration_stats: Record<string, DurationStat>
  by_red: Record<string, Record<string, number>>
  chi2_independence_test: Chi2Test
}

export interface GoldCase {
  case_id: string
  text: string
  final_level: number | null
  final_citation: string | null
  ambiguous: boolean
  should_abstain: boolean
  resolution_method: string
  nota_resolucion?: string
}

export interface CasosResponse {
  total: number
  cases: GoldCase[]
}

export interface CorpusChunk {
  chunk_id: string
  text: string
  source: string
  level: number | null
}

export interface CorpusResponse {
  corpus_format: CorpusFormat
  total: number
  chunks: CorpusChunk[]
}
