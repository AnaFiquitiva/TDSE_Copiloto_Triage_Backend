"""Interfaz gráfica del prototipo (Tkinter, sin dependencias externas).

Ofrece tres pestañas, cada una envolviendo un punto de entrada ya existente
del prototipo (no duplica lógica, solo la expone visualmente):

1. **Consulta individual**: equivalente grafico de `demo.py`. Muestra la
   sugerencia del copiloto (nivel + cita, o abstención), los fragmentos
   recuperados con su score, y la sugerencia de la línea base C0 para
   comparar.
2. **Experimento 2x2**: equivalente grafico de
   `python -m experiments.run_experiment`. Corre las celdas E1-E4 + C0 sobre
   `data/cases.json` y muestra la tabla de resultados.
3. **Dataset real**: equivalente grafico de
   `python -m experiments.analyze_external_data`. Corre el análisis del
   dataset real de Datos Abiertos Colombia y muestra el resumen.

Se ejecuta con:
    python gui.py

Cada acción pesada corre en un hilo en segundo plano para no congelar la
ventana; los widgets solo se actualizan desde el hilo principal (vía
`root.after`), como exige Tkinter.
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments import run_experiment  # noqa: E402
from experiments.analyze_external_data import (  # noqa: E402
    DATA_PATH as EXTERNAL_DATA_PATH,
)
from experiments.analyze_external_data import (  # noqa: E402
    level_distribution,
    load_rows,
    time_to_attention_minutes,
)
from experiments.analyze_external_data import format_markdown as format_external_markdown  # noqa: E402
from src.baseline import RuleBasedBaseline  # noqa: E402
from src.pipeline import CopilotoPipeline  # noqa: E402

DISCLAIMER = (
    "Este es un prototipo académico. La sugerencia asiste, no decide: la "
    "clasificación final es siempre responsabilidad del personal de salud."
)


def _run_in_background(widget: tk.Widget, target, on_done) -> None:
    """Corre `target()` en un hilo aparte y entrega su resultado (o la
    excepción) a `on_done`. `on_done` se agenda con `widget.after(0, ...)`
    para que se ejecute en el hilo principal de Tkinter, ya que actualizar
    widgets directamente desde un hilo secundario no es seguro."""

    def worker():
        try:
            result = target()
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario, no se oculta
            result = exc
        widget.after(0, lambda: on_done(result))

    threading.Thread(target=worker, daemon=True).start()


class ConsultaTab(ttk.Frame):
    """Pestaña de consulta individual: equivalente grafico de demo.py."""

    def __init__(self, master: ttk.Notebook):
        super().__init__(master, padding=10)
        self.baseline = RuleBasedBaseline()

        ttk.Label(self, text="Relato libre del paciente:").grid(row=0, column=0, sticky="w")
        self.texto = tk.Text(self, height=4, width=80, wrap="word")
        self.texto.grid(row=1, column=0, columnspan=4, sticky="we", pady=(0, 8))
        self.texto.insert(
            "1.0", "el paciente tiene dolor en el pecho que se corre al brazo y esta sudando frio"
        )

        ttk.Label(self, text="Corpus:").grid(row=2, column=0, sticky="e")
        self.corpus_var = tk.StringVar(value="reformatted")
        ttk.Combobox(
            self, textvariable=self.corpus_var, values=["raw", "reformatted"],
            state="readonly", width=15,
        ).grid(row=2, column=1, sticky="w")

        ttk.Label(self, text="Embeddings:").grid(row=2, column=2, sticky="e")
        self.embed_var = tk.StringVar(value="clinical_es")
        ttk.Combobox(
            self, textvariable=self.embed_var, values=["generic", "clinical_es"],
            state="readonly", width=15,
        ).grid(row=2, column=3, sticky="w")

        ttk.Label(self, text="Backend:").grid(row=3, column=0, sticky="e")
        self.backend_var = tk.StringVar(value="gemini")
        ttk.Combobox(
            self, textvariable=self.backend_var, values=["gemini", "deterministic"],
            state="readonly", width=15,
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(
            self, text="'gemini' requiere GEMINI_API_KEY (ver README); si falla, usa el respaldo (modo degradado).",
            foreground="#555555", font=("TkDefaultFont", 8),
        ).grid(row=3, column=2, columnspan=2, sticky="w")

        self.boton = ttk.Button(self, text="Obtener sugerencia", command=self._on_click)
        self.boton.grid(row=4, column=0, columnspan=4, pady=8)

        self.salida = tk.Text(self, height=22, width=100, wrap="word", state="disabled")
        self.salida.grid(row=5, column=0, columnspan=4, sticky="nsew")

        ttk.Label(self, text=DISCLAIMER, foreground="#8a1f1f", wraplength=760).grid(
            row=6, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )

        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

    def _write(self, text: str) -> None:
        self.salida.configure(state="normal")
        self.salida.delete("1.0", "end")
        self.salida.insert("1.0", text)
        self.salida.configure(state="disabled")

    def _on_click(self) -> None:
        relato = self.texto.get("1.0", "end").strip()
        if not relato:
            self._write("Escriba primero el relato del paciente.")
            return
        corpus_format = self.corpus_var.get()
        embedding_mode = self.embed_var.get()
        backend = self.backend_var.get()
        self.boton.configure(state="disabled")
        self._write("Procesando..." if backend == "deterministic" else "Procesando (llamando a Gemini)...")

        def compute():
            return self._compute(relato, corpus_format, embedding_mode, backend)

        def done(result):
            self._on_result(result, relato, corpus_format, embedding_mode, backend)

        _run_in_background(self, compute, done)

    def _compute(self, relato: str, corpus_format: str, embedding_mode: str, backend: str):
        """Lógica pura (sin Tkinter) de una consulta: separada de `_on_click`
        para poder probarla directamente, sin hilos ni mainloop."""
        import os
        import tempfile

        fd, tmp_name = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        audit_path = Path(tmp_name)
        try:
            pipeline = CopilotoPipeline(
                corpus_format=corpus_format,
                embedding_mode=embedding_mode,
                backend=backend,
                audit_path=audit_path,
            )
            suggestion = pipeline.run_case("gui", relato)
            backend_used = suggestion.backend_used
        finally:
            audit_path.unlink(missing_ok=True)
        baseline_result = self.baseline.classify(relato)
        return suggestion, baseline_result, backend_used

    def _on_result(self, result, relato: str, corpus_format: str, embedding_mode: str, backend: str) -> None:
        self.boton.configure(state="normal")
        if isinstance(result, Exception):
            self._write(f"Error: {result}")
            return
        suggestion, baseline_result, backend_used = result
        lines = [
            f"Relato: {relato}",
            f"Corpus={corpus_format} | Embeddings={embedding_mode} | Backend solicitado={backend} "
            f"| Backend usado={backend_used}",
            "-" * 70,
        ]
        if suggestion.degraded:
            lines.append(
                "[MODO DEGRADADO: se pidió la ruta 'gemini' pero no está disponible (revise "
                "GEMINI_API_KEY o la conexión); se usó el respaldo determinista.]"
            )
        if suggestion.abstained:
            lines.append(f"Copiloto: SE ABSTIENE (razón: {suggestion.reason})")
        else:
            lines.append(
                f"Copiloto: Nivel {suggestion.level} | Cita: {suggestion.citation} "
                f"| Confianza: {suggestion.confidence:.2f}"
            )
        lines.append("")
        lines.append("Fragmentos recuperados:")
        for rc in suggestion.retrieved:
            snippet = rc.chunk.text.replace("\n", " ")[:100]
            lines.append(f"  - {rc.chunk.chunk_id} (score={rc.score:.3f}): {snippet!r}")
        lines.append("-" * 70)
        lines.append(
            f"Línea base (reglas): Nivel {baseline_result.level} | "
            f"Cita: {baseline_result.citation} | Palabra clave: {baseline_result.matched_keyword}"
        )
        self._write("\n".join(lines))


class ExperimentoTab(ttk.Frame):
    """Pestaña del experimento 2x2: equivalente grafico de run_experiment.py."""

    def __init__(self, master: ttk.Notebook):
        super().__init__(master, padding=10)
        ttk.Label(
            self,
            text="Ejecuta las celdas E1-E4 y la línea base C0 sobre data/cases.json "
            "(el mismo comando que `python -m experiments.run_experiment`).",
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")

        self.boton = ttk.Button(self, text="Ejecutar experimento 2x2", command=self._on_click)
        self.boton.grid(row=1, column=0, pady=8, sticky="w")

        self.salida = tk.Text(self, height=28, width=110, wrap="word", state="disabled")
        self.salida.grid(row=2, column=0, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

    def _write(self, text: str) -> None:
        self.salida.configure(state="normal")
        self.salida.delete("1.0", "end")
        self.salida.insert("1.0", text)
        self.salida.configure(state="disabled")

    def _on_click(self) -> None:
        self.boton.configure(state="disabled")
        self._write("Ejecutando (puede tardar unos segundos)...")
        _run_in_background(self, self._compute, self._on_result)

    @staticmethod
    def _compute() -> str:
        """Lógica pura (sin Tkinter): el mismo camino que
        `python -m experiments.run_experiment`, separado de `_on_click` para
        poder probarlo directamente."""
        cases = run_experiment.load_cases()
        output = run_experiment.run_all(cases)
        return run_experiment.format_markdown(
            output["kappa_intra_equipo"], output["baseline_C0"], output["celdas"], n_total_cases=len(cases)
        )

    def _on_result(self, result) -> None:
        self.boton.configure(state="normal")
        if isinstance(result, Exception):
            self._write(f"Error: {result}")
            return
        self._write(result)


class DatasetTab(ttk.Frame):
    """Pestaña del dataset externo real: equivalente grafico de
    analyze_external_data.py."""

    def __init__(self, master: ttk.Notebook):
        super().__init__(master, padding=10)
        ttk.Label(
            self,
            text="Analiza data/external_triage_urgencias_colombia.csv "
            "(Datos Abiertos Colombia, dataset 'Clasificación en Triage Urgencias', "
            "~89.000 registros reales).",
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")

        self.boton = ttk.Button(self, text="Analizar dataset real", command=self._on_click)
        self.boton.grid(row=1, column=0, pady=8, sticky="w")

        self.salida = tk.Text(self, height=28, width=110, wrap="word", state="disabled")
        self.salida.grid(row=2, column=0, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

    def _write(self, text: str) -> None:
        self.salida.configure(state="normal")
        self.salida.delete("1.0", "end")
        self.salida.insert("1.0", text)
        self.salida.configure(state="disabled")

    def _on_click(self) -> None:
        self.boton.configure(state="disabled")
        self._write("Cargando y analizando ~89.000 registros...")
        _run_in_background(self, self._compute, self._on_result)

    @staticmethod
    def _compute() -> str:
        """Lógica pura (sin Tkinter): el mismo camino que
        `python -m experiments.analyze_external_data`, separado de
        `_on_click` para poder probarlo directamente."""
        if not EXTERNAL_DATA_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró {EXTERNAL_DATA_PATH}. Ver README.md, sección "
                "'Fuentes de datos reales'."
            )
        rows = load_rows()
        distribution = level_distribution(rows)
        durations = time_to_attention_minutes(rows)
        return format_external_markdown(rows, distribution, durations)

    def _on_result(self, result) -> None:
        self.boton.configure(state="normal")
        if isinstance(result, Exception):
            self._write(f"Error: {result}")
            return
        self._write(result)


def main() -> None:
    root = tk.Tk()
    root.title("TDSE · Copiloto de Triage RAG")
    root.geometry("900x650")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    notebook.add(ConsultaTab(notebook), text="Consulta individual")
    notebook.add(ExperimentoTab(notebook), text="Experimento 2x2")
    notebook.add(DatasetTab(notebook), text="Dataset real")

    root.mainloop()


if __name__ == "__main__":
    main()
