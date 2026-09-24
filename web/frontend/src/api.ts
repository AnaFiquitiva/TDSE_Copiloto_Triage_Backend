import type {
  Backend,
  CasosResponse,
  CorpusFormat,
  CorpusResponse,
  DatasetResponse,
  EmbeddingMode,
  ExperimentoResponse,
  ConsultaResponse,
  DecisionRequest,
  DecisionResponse,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }))
    // FastAPI devuelve `detail` como texto (HTTPException) o como lista de
    // errores de validación (422 de Pydantic).
    const detail = Array.isArray(body.detail)
      ? body.detail.map((d: { msg?: string }) => d.msg ?? String(d)).join('; ')
      : body.detail
    throw new Error(detail ?? `Error ${resp.status}`)
  }
  return resp.json() as Promise<T>
}

export function fetchConsulta(params: {
  texto: string
  corpus_format: CorpusFormat
  embedding_mode: EmbeddingMode
  backend: Backend
}): Promise<ConsultaResponse> {
  return request<ConsultaResponse>('/api/consulta', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function postDecision(body: DecisionRequest): Promise<DecisionResponse> {
  return request<DecisionResponse>('/api/decision', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function fetchExperimento(): Promise<ExperimentoResponse> {
  return request<ExperimentoResponse>('/api/experimento')
}

export function fetchDataset(): Promise<DatasetResponse> {
  return request<DatasetResponse>('/api/dataset')
}

export function fetchCasos(): Promise<CasosResponse> {
  return request<CasosResponse>('/api/casos')
}

export function fetchCorpus(corpusFormat: CorpusFormat): Promise<CorpusResponse> {
  return request<CorpusResponse>(`/api/corpus?corpus_format=${corpusFormat}`)
}
