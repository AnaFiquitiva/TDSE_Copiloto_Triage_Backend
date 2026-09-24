import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, ArrowUpCircle, BellRing, CheckCircle2, ChevronDown, ClipboardCheck, FileText, Send, ShieldAlert } from 'lucide-react'
import clsx from 'clsx'

import { fetchConsulta, postDecision } from '../api'
import type { ConsultaResponse, DecisionAction, DecisionResponse } from '../types'
import { Card, ErrorBox, ScoreBar, SectionTitle, Spinner } from '../components/Common'
import { LevelBadge, levelColor } from '../components/LevelBadge'

const FIXED_NOTICE = 'Sugerencia de apoyo. La clasificación final es responsabilidad del profesional de salud.'
const LEVEL_OPTIONS = [
  { value: 1, label: 'I · Resucitación' },
  { value: 2, label: 'II · Emergencia' },
  { value: 3, label: 'III · Urgencia' },
  { value: 4, label: 'IV · Urgencia menor' },
  { value: 5, label: 'V · No urgencia' },
]

export function RevisionView() {
  const [texto, setTexto] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ConsultaResponse | null>(null)

  async function onSubmit() {
    if (!texto.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      // Ruta principal por defecto (Gemini); el backend marca el modo degradado si cae al respaldo.
      setResult(
        await fetchConsulta({ texto, corpus_format: 'reformatted', embedding_mode: 'clinical_es', backend: 'gemini' }),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-4xl">
      <SectionTitle
        eyebrow="Componente 4 · Revisión humana"
        title="Revisión de la sugerencia"
        description="El profesional revisa la sugerencia, su cita y las alarmas detectadas, y registra una decisión explícita: confirmar el nivel o corregirlo con un motivo. Cada decisión queda en el registro de auditoría encadenado."
      />

      <p className="mb-5 flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm font-medium text-amber-200">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
        {FIXED_NOTICE}
      </p>

      <Card>
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-slate-400">Relato libre del paciente</span>
          <textarea
            value={texto}
            onChange={(e) => setTexto(e.target.value)}
            rows={3}
            className="resize-none rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-2.5 text-sm leading-relaxed text-slate-100 outline-none ring-teal-500/50 placeholder:text-slate-500 focus:ring-2"
            placeholder="Ej: dice que ya tiene todo listo para hacerse daño esta noche..."
          />
        </label>
        <button
          onClick={onSubmit}
          disabled={loading || !texto.trim()}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-teal-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-teal-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Send className="h-4 w-4" />
          {loading ? 'Procesando...' : 'Obtener sugerencia'}
        </button>
      </Card>

      <div className="mt-6">
        {loading && <Spinner label="Consultando la ruta principal..." />}
        {error && <ErrorBox message={error} />}
        {/* key: una tarjeta nueva por consulta, para reiniciar el cronómetro y la elección. */}
        {!loading && result && <SuggestionReview key={result.record_hash} result={result} />}
      </div>
    </div>
  )
}

function SuggestionReview({ result }: { result: ConsultaResponse }) {
  const shownAt = useRef<number>(0)
  useEffect(() => {
    shownAt.current = performance.now()
  }, [])

  const { suggestion, n_final, n_modelo, n_reglas, alarms } = result
  const hasLevel = n_final !== null
  const raisedByRule = hasLevel && n_modelo !== null && n_reglas !== null && n_reglas < n_modelo
  const ruleReference = !hasLevel && n_reglas !== null ? alarms.find((a) => a.level === n_reglas) : undefined

  return (
    <div className="flex flex-col gap-5">
      {(result.degraded || result.escalated) && (
        <div className="flex flex-col gap-2">
          {result.degraded && (
            <Banner tone="amber" icon={AlertTriangle}>
              Modo degradado: respaldo determinista (la ruta principal Gemini no estuvo disponible).
            </Banner>
          )}
          {result.escalated && (
            <Banner tone="red" icon={BellRing}>
              Escalar a revisión prioritaria: el sistema se abstuvo y no propone un nivel.
            </Banner>
          )}
        </div>
      )}

      <Card className="flex flex-col gap-4">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
          <ClipboardCheck className="h-4 w-4 text-teal-400" />
          Sugerencia
        </h3>

        {hasLevel ? (
          <div className="flex flex-wrap items-center gap-3">
            <LevelBadge level={n_final} />
            <code className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300">{suggestion.citation}</code>
            {raisedByRule && (
              <span className="flex items-center gap-1 rounded-full bg-red-500/15 px-2.5 py-1 text-[11px] font-medium text-red-300">
                <ArrowUpCircle className="h-3.5 w-3.5" />
                Elevado por regla de alarma (el modelo sugería nivel {n_modelo}, cita {result.model_citation})
              </span>
            )}
          </div>
        ) : (
          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4 text-sm text-slate-300">
            <p className="font-semibold">Sin sugerencia del modelo (abstención: {suggestion.reason}).</p>
            {ruleReference && (
              <p className="mt-2 text-xs text-slate-400">
                Referencia de una regla de alarma, no del modelo: nivel {ruleReference.level} (
                <code>{ruleReference.citation}</code>).
              </p>
            )}
          </div>
        )}

        {result.cited_fragment_text && (
          <div>
            <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
              <FileText className="h-3.5 w-3.5" />
              Fragmento citado (texto íntegro)
            </p>
            <pre className="whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-950/50 p-3 font-sans text-xs leading-relaxed text-slate-300">
              {result.cited_fragment_text}
            </pre>
          </div>
        )}

        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Alarmas detectadas</p>
          {alarms.length === 0 ? (
            <p className="text-xs text-slate-500">Ninguna.</p>
          ) : (
            <ul className="flex flex-col gap-1.5">
              {alarms.map((a) => (
                <li key={a.rule_id} className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
                  <LevelBadge level={a.level} compact />
                  <code className="text-slate-300">{a.rule_id}</code>
                  <span>
                    “{a.matched_pattern}” → <code>{a.citation}</code>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <details className="group rounded-xl border border-slate-800 bg-slate-950/30 p-3">
          <summary className="flex cursor-pointer list-none items-center justify-between text-xs font-semibold text-slate-400">
            Fragmentos recuperados ({result.retrieved.length})
            <ChevronDown className="h-4 w-4 transition group-open:rotate-180" />
          </summary>
          <ul className="mt-3 flex flex-col gap-2.5">
            {result.retrieved.map((rc) => (
              <li key={rc.chunk_id}>
                <div className="mb-1 flex justify-between gap-2 text-xs">
                  <code className={clsx(rc.chunk_id === suggestion.citation ? 'text-teal-300' : 'text-slate-300')}>
                    {rc.chunk_id}
                  </code>
                  <span className="text-slate-500">{rc.score.toFixed(4)}</span>
                </div>
                <ScoreBar value={rc.score} max={Math.max(result.retrieved[0]?.score ?? 1, 1e-6)} color={levelColor(rc.level)} />
              </li>
            ))}
          </ul>
        </details>

        <p className="text-[11px] text-slate-500">
          Ruta: <code>{result.backend_used}</code> · registro <code>{result.record_hash.slice(0, 12)}…</code>
        </p>
      </Card>

      <DecisionForm result={result} hasLevel={hasLevel} shownAt={shownAt} />
    </div>
  )
}

function DecisionForm({
  result,
  hasLevel,
  shownAt,
}: {
  result: ConsultaResponse
  hasLevel: boolean
  shownAt: React.RefObject<number>
}) {
  // Sin opción preseleccionada: la confirmación exige una elección activa.
  const [action, setAction] = useState<DecisionAction | null>(null)
  const [level, setLevel] = useState<number | null>(null)
  const [reason, setReason] = useState('')
  const [user, setUser] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState<DecisionResponse | null>(null)

  const valid =
    (action === 'confirma' && hasLevel) || (action === 'corrige' && level !== null && reason.trim().length > 0)

  async function onRegister() {
    if (!valid || action === null) return
    setSending(true)
    setError(null)
    try {
      const reviewSeconds = (performance.now() - (shownAt.current ?? performance.now())) / 1000
      setDone(
        await postDecision({
          record_hash: result.record_hash,
          action,
          final_level: action === 'corrige' ? (level ?? undefined) : undefined,
          reason: action === 'corrige' ? reason.trim() : undefined,
          review_seconds: Math.round(reviewSeconds * 10) / 10,
          user: user.trim() || undefined,
        }),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error desconocido')
    } finally {
      setSending(false)
    }
  }

  if (done) {
    return (
      <Card className="flex items-start gap-3 border-emerald-500/30">
        <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
        <div className="text-sm text-slate-300">
          <p className="font-semibold text-emerald-300">
            Decisión registrada: {done.action === 'confirma' ? 'confirma' : 'corrige'} → nivel {done.final_level}
          </p>
          <p className="mt-1 break-all text-xs text-slate-500">
            Hash del evento: <code className="text-slate-400">{done.event_hash}</code>
          </p>
        </div>
      </Card>
    )
  }

  return (
    <Card className="flex flex-col gap-4">
      <h3 className="text-sm font-semibold text-slate-200">Decisión del profesional</h3>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2" role="radiogroup" aria-label="Decisión">
        <ChoiceCard
          selected={action === 'confirma'}
          disabled={!hasLevel}
          onSelect={() => setAction('confirma')}
          title="Confirmar nivel"
          hint={hasLevel ? `Acepto el nivel ${result.n_final} sugerido.` : 'No hay nivel que confirmar (abstención).'}
        />
        <ChoiceCard
          selected={action === 'corrige'}
          onSelect={() => setAction('corrige')}
          title="Corregir nivel"
          hint="Elijo otro nivel y explico por qué."
        />
      </div>

      {action === 'corrige' && (
        <div className="flex flex-col gap-3">
          <label className="flex flex-col gap-1.5 text-xs font-medium text-slate-400">
            Nivel final
            <select
              value={level ?? ''}
              onChange={(e) => setLevel(e.target.value ? Number(e.target.value) : null)}
              className="rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-2 text-sm text-slate-100 outline-none ring-teal-500/50 focus:ring-2"
            >
              <option value="">Seleccione un nivel…</option>
              {LEVEL_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1.5 text-xs font-medium text-slate-400">
            Motivo de la corrección (obligatorio)
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={2}
              className="resize-none rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-2 text-sm text-slate-100 outline-none ring-teal-500/50 focus:ring-2"
            />
          </label>
        </div>
      )}

      <label className="flex flex-col gap-1.5 text-xs font-medium text-slate-400">
        Usuario (opcional)
        <input
          value={user}
          onChange={(e) => setUser(e.target.value)}
          className="rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-2 text-sm text-slate-100 outline-none ring-teal-500/50 focus:ring-2"
        />
      </label>

      {error && <ErrorBox message={error} />}

      <button
        onClick={onRegister}
        disabled={!valid || sending}
        className="flex items-center justify-center gap-2 rounded-lg bg-teal-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-teal-400 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {sending ? 'Registrando...' : 'Registrar decisión'}
      </button>
    </Card>
  )
}

function ChoiceCard({
  selected,
  disabled = false,
  onSelect,
  title,
  hint,
}: {
  selected: boolean
  disabled?: boolean
  onSelect: () => void
  title: string
  hint: string
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      disabled={disabled}
      onClick={onSelect}
      className={clsx(
        'rounded-xl border p-3 text-left transition disabled:cursor-not-allowed disabled:opacity-40',
        selected
          ? 'border-teal-500/60 bg-teal-500/10 ring-1 ring-inset ring-teal-500/40'
          : 'border-slate-700 bg-slate-800/40 hover:border-slate-500',
      )}
    >
      <p className={clsx('text-sm font-semibold', selected ? 'text-teal-300' : 'text-slate-200')}>{title}</p>
      <p className="mt-0.5 text-xs text-slate-500">{hint}</p>
    </button>
  )
}

function Banner({
  tone,
  icon: Icon,
  children,
}: {
  tone: 'amber' | 'red'
  icon: typeof AlertTriangle
  children: React.ReactNode
}) {
  return (
    <div
      className={clsx(
        'flex items-start gap-3 rounded-xl border px-4 py-3 text-sm font-medium',
        tone === 'amber' ? 'border-amber-500/30 bg-amber-500/10 text-amber-200' : 'border-red-500/40 bg-red-500/10 text-red-200',
      )}
    >
      <Icon className="mt-0.5 h-5 w-5 shrink-0" />
      <p>{children}</p>
    </div>
  )
}
