"""Motor de decisiones: traduce los indicadores en acciones concretas, cada una con su justificación.

Las reglas son deliberadamente simples y auditables (no una caja negra), para que un gestor pueda discutirlas:

  Prioridad 1 · ACTUAR YA   : municipio de nivel de riesgo ALTO y con actividad sísmica inusual en los últimos 12 meses.
  Prioridad 2 · PREPARAR    : municipio de nivel de riesgo ALTO con actividad normal (el riesgo viene de la amenaza,
                              la población y el terreno, no de un repunte reciente).
  Prioridad 3 · VIGILAR     : actividad inusual en un municipio de nivel Medio/Bajo (señal para observar, no para alarmar).
  Sin acción extraordinaria : el resto, que continúa con seguimiento de rutina.

Además, si la vulnerabilidad del terreno está entre el 10 % más alto del departamento se añade estudiar su microzonificación.
El plan respeta un presupuesto de evaluación (en unidades SIMULADAS, las mismas del agente de refuerzo): se cubren los
municipios por orden de prioridad e IRI hasta agotarlo.

Son recomendaciones de apoyo a la decisión; las umbrales los propone el equipo y deben validarlos especialistas.
"""
import pandas as pd

from .reinforcement import PRESUPUESTO, costo_simulado

ACCIONES = {
    1: ("Actuar ya", "Inspección estructural prioritaria de hospitales, colegios y puentes, y refuerzo del monitoreo sísmico."),
    2: ("Preparar", "Evaluar la vulnerabilidad de edificaciones esenciales, actualizar el plan municipal de gestión del riesgo y programar simulacros."),
    3: ("Vigilar", "Seguimiento semanal de la actividad y verificación de la red de monitoreo; sin medidas extraordinarias."),
}


def _miles(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")  # formato colombiano: 617.659


def recomendar(tabla: pd.DataFrame, presupuesto: float = PRESUPUESTO) -> pd.DataFrame:
    """`tabla` trae IRI, nivel, semáforo, población, PGA, vulnerabilidad, actividad de 12 meses, área y distancia a Bucaramanga."""
    t = tabla.copy()
    total_pob = t["poblacion"].sum()
    corte_vuln = t["vulnerabilidad"].quantile(0.90)
    inusual = t["semaforo"] != "Verde"
    t["prioridad"] = 0
    t.loc[(t["nivel"] == "Alto") & ~inusual, "prioridad"] = 2
    t.loc[(t["nivel"] == "Alto") & inusual, "prioridad"] = 1
    t.loc[(t["nivel"] != "Alto") & inusual, "prioridad"] = 3
    r = t[t["prioridad"] > 0].copy()

    def porque(f):
        base = (f"Nivel {f['nivel']} (IRI {f['iri']:.0f}/100, puesto {int(f['ranking'])} de {len(t)}): viven {_miles(f['poblacion'])} personas "
                f"({f['poblacion'] / total_pob:.1%} de Santander), PGA esperado {f['PGA_medio']:.2f} g y vulnerabilidad del sitio {f['vulnerabilidad']:.2f}. "
                f"En los últimos 12 meses registró {int(f['observados_12m'])} sismos M≥4 frente a {f['esperados_12m']:.1f} esperados ({f['semaforo']}).")
        return base

    def accion(f):
        txt = ACCIONES[int(f["prioridad"])][1]
        if f["vulnerabilidad"] >= corte_vuln:
            txt += " Además, estudiar la microzonificación de sus suelos."
        return txt

    r["decision"] = r["prioridad"].map(lambda p: ACCIONES[int(p)][0])
    r["accion"] = r.apply(accion, axis=1)
    r["porque"] = r.apply(porque, axis=1)
    r["costo_simulado"] = costo_simulado(r["area_km2"], r["distancia_bucaramanga_km"]).round(1)

    r = r.sort_values(["prioridad", "iri"], ascending=[True, False]).reset_index(drop=True)
    acumulado, en_plan = 0.0, []
    for c in r["costo_simulado"]:
        if acumulado + c <= presupuesto:
            acumulado += c
            en_plan.append(True)
        else:
            en_plan.append(False)
    r["en_plan"] = en_plan
    r.attrs.update({"presupuesto": presupuesto, "costo_plan": round(acumulado, 1), "poblacion_total": int(total_pob)})
    return r[["prioridad", "decision", "municipio", "accion", "porque", "poblacion", "iri", "costo_simulado", "en_plan"]]


def resumen(reco: pd.DataFrame) -> str:
    """Titular de la decisión, en lenguaje llano."""
    n = reco["prioridad"].value_counts().to_dict()
    plan = reco[reco["en_plan"]]
    cubierta = plan["poblacion"].sum()
    partes = []
    if n.get(1):
        partes.append(f"**actuar ya** en {n[1]} municipio{'s' if n[1] > 1 else ''} ({', '.join(reco[reco['prioridad'] == 1]['municipio'])})")
    if n.get(2):
        partes.append(f"**preparar** {n[2]} más de nivel Alto")
    if n.get(3):
        partes.append(f"**vigilar** {n[3]} con actividad inusual ({', '.join(reco[reco['prioridad'] == 3]['municipio'].head(3))})")
    texto = "Se recomienda " + ", ".join(partes) + "." if partes else "No hay municipios que requieran acciones extraordinarias."
    texto += (f" Con el presupuesto simulado de {reco.attrs['presupuesto']:.0f} unidades se puede evaluar a {len(plan)} municipios "
              f"({_miles(cubierta)} habitantes, {cubierta / reco.attrs['poblacion_total']:.0%} de Santander), en ese orden de prioridad.")
    return texto
