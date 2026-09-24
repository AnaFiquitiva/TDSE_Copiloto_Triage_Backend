import { useState } from 'react'
import { BookOpen, ClipboardCheck, Database, ExternalLink, FlaskConical, MessageSquareText, Stethoscope } from 'lucide-react'
import clsx from 'clsx'

import { ConsultaView } from './views/ConsultaView'
import { ExperimentoView } from './views/ExperimentoView'
import { DatasetView } from './views/DatasetView'
import { ExplorarView } from './views/ExplorarView'
import { RevisionView } from './views/RevisionView'

type Tab = 'revision' | 'consulta' | 'experimento' | 'dataset' | 'explorar'

const NAV_ITEMS: { id: Tab; label: string; icon: typeof MessageSquareText }[] = [
  { id: 'revision', label: 'Revisión', icon: ClipboardCheck },
  { id: 'consulta', label: 'Consulta individual', icon: MessageSquareText },
  { id: 'experimento', label: 'Experimento 2x2', icon: FlaskConical },
  { id: 'dataset', label: 'Dataset real', icon: Database },
  { id: 'explorar', label: 'Corpus y casos', icon: BookOpen },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('consulta')

  return (
    <div className="flex min-h-screen bg-slate-950">
      {/* Fondo decorativo con gradiente sutil, para el efecto "llamativo" sin distraer */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-40 -top-40 h-96 w-96 rounded-full bg-teal-500/10 blur-3xl" />
        <div className="absolute -bottom-40 -right-40 h-96 w-96 rounded-full bg-sky-500/10 blur-3xl" />
      </div>

      <aside className="relative z-10 flex w-64 shrink-0 flex-col border-r border-slate-800/80 bg-slate-950/80 backdrop-blur">
        <div className="flex items-center gap-2.5 border-b border-slate-800/80 px-5 py-5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-teal-400 to-sky-500 text-slate-950">
            <Stethoscope className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-bold leading-tight text-slate-50">TDSE</p>
            <p className="text-[11px] leading-tight text-slate-500">Copiloto de Triage RAG</p>
          </div>
        </div>

        <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => setTab(item.id)}
              className={clsx(
                'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition',
                tab === item.id
                  ? 'bg-teal-500/15 text-teal-300 ring-1 ring-inset ring-teal-500/30'
                  : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200',
              )}
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="border-t border-slate-800/80 px-5 py-4">
          <a
            href="https://github.com/AnaFiquitiva/TDSE_Copiloto_de_Triage_RAG"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 text-xs text-slate-500 transition hover:text-slate-300"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            Ver repositorio
          </a>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-600">
            Prototipo académico TDSE. Asiste, no decide.
          </p>
        </div>
      </aside>

      <main className="relative z-10 flex-1 overflow-y-auto px-6 py-8 sm:px-10">
        {tab === 'revision' && <RevisionView />}
        {tab === 'consulta' && <ConsultaView />}
        {tab === 'experimento' && <ExperimentoView />}
        {tab === 'dataset' && <DatasetView />}
        {tab === 'explorar' && <ExplorarView />}
      </main>
    </div>
  )
}
