# Revisión de las advertencias del validador (C003, C004, C006, C045, C058)

Generado a partir de `python -m experiments.validate_cases` y de las
predicciones de `results/raw_results.json`. **No se modificó
`data/cases.json`**: los niveles, citas y notas son del equipo. Este documento
solo presenta la evidencia y las opciones; la decisión es del equipo.

## C003 — acuerdo en el nivel, citas distintas (requiere decisión)

> Tiene una herida profunda en el brazo con un sangrado que no cede aunque le
> hacemos presion directa; ya se ve muy palido, mareado y casi no puede hablar.

| | Nivel | Cita |
|---|---|---|
| Evaluador 1 | I | `ALARMA-SANGRADO-GRAVE` |
| Evaluador 2 | I | `RES5596-N1` |
| Final | I | `RES5596-N1` (`acuerdo_inicial`, sin nota) |

**Evidencia del corpus.** Las dos citas son de nivel I y ambas son
defendibles:

- `ALARMA-SANGRADO-GRAVE`: "Hemorragia que no cede con presión directa o hay
  signos de shock → Nivel I". Coincide casi literalmente con el relato
  ("no cede aunque le hacemos presión directa", palidez y mareo).
- `RES5596-N1`: criterios generales de nivel I, entre ellos "hemorragia masiva
  no controlada" y "shock".

**Por qué importa.** El nivel no está en discusión, pero la cita de referencia
sí afecta las métricas de recuperación y de fidelidad. E5 elige
`ALARMA-SANGRADO-GRAVE`, y `RES5596-N1` no aparece en su top-5. Por eso C003
cuenta como fallo de recall@k y de fidelidad en E5, aunque la sugerencia es
correcta y cita el criterio más específico. E4 también recupera
`ALARMA-SANGRADO-GRAVE` en el primer lugar.

**Opciones para el equipo:**

1. **Mantener `RES5596-N1`** y documentar en `nota_resolucion` por qué se
   prefirió el criterio general.
2. **Cambiar a `ALARMA-SANGRADO-GRAVE`.** Como los evaluadores discreparon en
   la cita, el método debería pasar a `consenso_tras_discusion`, con su
   `nota_resolucion`. Efecto estimado: +1 caso en recall@k de E4 y E5 y +1 en
   la fidelidad de E5.
3. **Aceptar varias citas válidas por caso.** Por ejemplo, un campo
   `acceptable_citations`. Es un cambio al esquema del paper y a las métricas,
   no solo a este caso.

Sea cual sea la decisión, conviene tomarla **antes** de mirar más resultados y
aplicarla con el mismo criterio a cualquier otro caso donde un criterio
general y uno específico coincidan.

## C004, C006, C045, C058 — el evaluador 2 asigna nivel II citando un criterio de nivel I

| Caso | Criterio citado por el evaluador 2 (nivel I) | Evaluador 2 | Resolución | Final | ¿Documentado? |
|---|---|---|---|---|---|
| C004 | `ALARMA-FIEBRE-GRAVE` | II | consenso | I | Sí (`nota_resolucion`) |
| C045 | `ALARMA-INTOXICACION-GRAVE` | II | consenso | I | Sí (`nota_resolucion`) |
| C006 | `ALARMA-TRAUMA-GRAVE` | II | desempate (evaluador 3: I) | I | **No hay nota** |
| C058 | `ALARMA-TRAUMA-GRAVE` | II | desempate (evaluador 3: I) | I | **No hay nota** |

**Diagnóstico.**

- **El nivel final de los cuatro casos es consistente:** la cita final es de
  nivel I y el nivel final es I. No hay error en la referencia que usan las
  métricas.
- **La contradicción está en la anotación ciega original del evaluador 2.**
  Identificó el criterio correcto, pero le asignó un nivel menos. **No debe
  corregirse:** es el registro de la discrepancia que después se resolvió, y
  cambiarlo alteraría el kappa intra-equipo y falsearía el proceso de doble
  ciego.
- **Hay un patrón.** 4 de las 9 discrepancias de nivel del gold standard
  tienen esta forma: el evaluador 2 cita la rama GRAVE de un flujograma y
  sub-clasifica un nivel. Es un sesgo sistemático de sub-triage del evaluador
  (confundir "emergencia" con "resucitación"), justo el tipo de error que el
  proyecto quiere medir. El sesgo es más amplio que estos cuatro casos: en
  **8 de las 9** discrepancias de nivel, el evaluador 2 es el menos urgente
  (C004, C006, C020, C045, C049, C058, C061, C072; solo C015 va en sentido
  contrario). Vale la pena reportarlo en el paper como hallazgo del
  proceso de etiquetado, y recalibrar al evaluador antes de etiquetar
  `data/cases_nuevos.json`.

**Acciones sugeridas para el equipo:**

1. **C006 y C058:** agregar `nota_resolucion` con la justificación real del
   desempate. En ambos, `ALARMA-TRAUMA-GRAVE` lista "pupilas asimétricas" como
   criterio de nivel I, y C058 además tiene pérdida de conciencia prolongada.
   La nota debe escribirla quien participó en el desempate: no se redactó
   aquí para no inventar el registro.
2. **C004 y C045:** no requieren cambios.
3. **Para los casos nuevos:** antes del doble ciego, repasar entre los
   evaluadores que cada rama `ALARMA-*-GRAVE` ya fija su nivel en el corpus.
   Así se separa la discrepancia clínica real del error de aplicar el criterio.
