import { useState } from 'react'
import { AlertOctagon, FileText, Send, Sparkles } from 'lucide-react'
import clsx from 'clsx'

import { fetchConsulta } from '../api'
import type { Backend, ConsultaResponse, CorpusFormat, EmbeddingMode } from '../types'
import { Card, DisclaimerBanner, ErrorBox, ScoreBar, SectionTitle, Spinner } from '../components/Common'
import { LevelBadge, levelColor } from '../components/LevelBadge'

const EJEMPLOS = [
  'el paciente tiene dolor en el pecho que se corre al brazo y esta sudando frio',
  'el nino tiene mucha fiebre, esta muy decaido, casi no reacciona y tiene el cuello rigido',
  'se quemo la mano con la plancha, es una quemadura pequena y superficial',
  'vengo a pedir un certificado medico para el trabajo',
  'no me siento bien desde hace unos dias, algo me pasa pero no sabria decir que es',
]

function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  return (
    <label className="flex flex-col gap-1.5 text-xs font-medium text-slate-400">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-2 text-sm text-slate-100 outline-none ring-teal-500/50 focus:ring-2"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  )
}

export function ConsultaView() {
  const [texto, setTexto] = useState(EJEMPLOS[0])
  const [corpusFormat, setCorpusFormat] = useState<CorpusFormat>('reformatted')
  const [embeddingMode, setEmbeddingMode] = useState<EmbeddingMode>('clinical_es')
  const [backend, setBackend] = useState<Backend>('gemini')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ConsultaResponse | null>(null)

  async function onSubmit() {
    if (!texto.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetchConsulta({ texto, corpus_format: corpusFormat, embedding_mode: embeddingMode, backend })
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-5xl">
      <SectionTitle
        eyebrow="Componente 3 · Generación"
        title="Consulta individual"
        description="Escribe el relato libre de un paciente y observa la sugerencia del copiloto (nivel + cita, o abstención), la evidencia recuperada, y la comparación con la línea base de reglas."
      />

      <Card>
        <div className="flex flex-col gap-4">
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-slate-400">Relato libre del paciente</span>
            <textarea
              value={texto}
              onChange={(e) => setTexto(e.target.value)}
              rows={3}
              className="resize-none rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-2.5 text-sm leading-relaxed text-slate-100 outline-none ring-teal-500/50 placeholder:text-slate-500 focus:ring-2"
              placeholder="Ej: el paciente tiene dolor en el pecho y sudoracion fria..."
            />
          </label>

          <div className="flex flex-wrap gap-2">
            {EJEMPLOS.map((ej, i) => (
              <button
                key={i}
                onClick={() => setTexto(ej)}
                className="rounded-full border border-slate-700 bg-slate-800/50 px-3 py-1 text-xs text-slate-400 transition hover:border-teal-500/50 hover:text-teal-300"
              >
                Ejemplo {i + 1}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <Select
              label="Corpus"
              value={corpusFormat}
              onChange={setCorpusFormat}
              options={[
                { value: 'reformatted', label: 'Reformateado' },
                { value: 'raw', label: 'Crudo' },
              ]}
            />
            <Select
              label="Embeddings"
              value={embeddingMode}
              onChange={setEmbeddingMode}
              options={[
                { value: 'clinical_es', label: 'Clínico ES (léxico)' },
                { value: 'generic', label: 'Genérico (TF-IDF)' },
              ]}
            />
            <Select
              label="Backend"
              value={backend}
              onChange={setBackend}
              options={[
                { value: 'gemini', label: 'Gemini (ruta principal)' },
                { value: 'deterministic', label: 'Determinista (sin red)' },
              ]}
            />
          </div>

          {backend === 'gemini' && (
            <p className="text-xs text-slate-500">
              Requiere <code className="rounded bg-slate-800 px-1 py-0.5">GEMINI_API_KEY</code> en el
              backend; si no está disponible, se usa el respaldo determinista y la respuesta queda marcada como modo degradado.
            </p>
          )}

          <button
            onClick={onSubmit}
            disabled={loading || !texto.trim()}
            className="flex items-center justify-center gap-2 rounded-lg bg-teal-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-teal-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
            {loading ? 'Procesando...' : 'Obtener sugerencia'}
          </button>
        </div>
      </Card>

      <div className="mt-6">
        {loading && <Spinner label={backend === 'gemini' ? 'Llamando a Gemini...' : 'Ejecutando el pipeline...'} />}
        {error && <ErrorBox message={error} />}
        {!loading && result && <ResultPanel result={result} />}
      </div>

      <div className="mt-8">
        <DisclaimerBanner />
      </div>
    </div>
  )
}

function ResultPanel({ result }: { result: ConsultaResponse }) {
  const { suggestion, retrieved, baseline, backend_used } = result
  const topScore = retrieved[0]?.score ?? 1

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
      <Card className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Sparkles className="h-4 w-4 text-teal-400" />
            Sugerencia del copiloto
          </h3>
          {backend_used === 'deterministic_fallback' && (
            <span className="rounded-full bg-amber-500/15 px-2.5 py-1 text-[11px] font-medium text-amber-300">
              Modo degradado: respaldo determinista
            </span>
          )}
        </div>

        {suggestion.abstained ? (
          <div className="flex items-start gap-3 rounded-xl border border-slate-700 bg-slate-800/60 p-4">
            <AlertOctagon className="mt-0.5 h-5 w-5 shrink-0 text-slate-400" />
            <div>
              <p className="font-semibold text-slate-200">Se abstiene</p>
              <p className="mt-1 text-xs text-slate-500">
                Razón: <code className="text-slate-400">{suggestion.reason}</code>
              </p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <LevelBadge level={suggestion.level} />
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <FileText className="h-4 w-4 text-slate-500" />
              <code className="rounded bg-slate-800 px-2 py-0.5 text-xs">{suggestion.citation}</code>
            </div>
            <div>
              <div className="mb-1 flex justify-between text-xs text-slate-500">
                <span>Confianza</span>
                <span>{(suggestion.confidence * 100).toFixed(0)}%</span>
              </div>
              <ScoreBar value={suggestion.confidence} color={levelColor(suggestion.level)} />
            </div>
          </div>
        )}

        <div className="border-t border-slate-800 pt-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Línea base (reglas, sin RAG)
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <LevelBadge level={baseline.level} compact />
            <code className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">{baseline.citation}</code>
            {baseline.matched_keyword && (
              <span className="text-xs text-slate-500">
                palabra clave: <em className="text-slate-400">“{baseline.matched_keyword}”</em>
              </span>
            )}
          </div>
        </div>
      </Card>

      <Card>
        <h3 className="mb-4 text-sm font-semibold text-slate-200">
          Fragmentos recuperados ({retrieved.length})
        </h3>
        <ul className="flex flex-col gap-3">
          {retrieved.map((rc) => (
            <li key={rc.chunk_id} className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
              <div className="mb-1.5 flex items-center justify-between gap-2">
                <code
                  className={clsx(
                    'truncate text-xs font-semibold',
                    rc.chunk_id === result.suggestion.citation ? 'text-teal-300' : 'text-slate-300',
                  )}
                >
                  {rc.chunk_id}
                </code>
                <span className="shrink-0 text-xs text-slate-500">{rc.score.toFixed(3)}</span>
              </div>
              <ScoreBar value={rc.score} max={Math.max(topScore, 0.001)} color={levelColor(rc.level)} />
              <p className="mt-2 line-clamp-2 text-xs text-slate-500">
                {rc.text.replace(/[#*[\]]/g, '').replace(/CITA:.*?\)/, '').trim()}
              </p>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
