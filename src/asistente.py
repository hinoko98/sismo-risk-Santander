"""Asistente de decisión: responde preguntas en lenguaje natural usando SOLO los datos calculados por el proyecto.

No es un modelo generativo: reconoce la intención de la pregunta (ranking, comparación, perfil de un municipio, alertas,
plan con presupuesto, método) y arma la respuesta con las tablas del Centro de decisión. Por eso funciona sin internet,
es reproducible y no puede inventar cifras: cada número que dice proviene de los datos.
"""
import difflib
import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decision import ACCIONES, recomendar, resumen


@dataclass
class Contexto:
    tabla: pd.DataFrame           # IRI + semáforo + prioridad del agente RL (una fila por municipio)
    reco: pd.DataFrame            # decisiones recomendadas
    proy: pd.DataFrame | None     # proyección Poisson (municipio, anio, p_acum_m4, ...)
    rf: dict                      # municipio -> categoría esperada por el Random Forest
    grupos: dict                  # municipio -> grupo K-Means
    anio_ini: int
    anio_fin: int
    anio_siguiente: int


def miles(n) -> str:
    return f"{int(round(float(n))):,}".replace(",", ".")


def dec(x, nd: int = 1) -> str:
    """Decimal colombiano: 0.26 -> '0,26'."""
    return f"{float(x):.{nd}f}".replace(".", ",")


def pct(x, nd: int = 0) -> str:
    return f"{float(x) * 100:.{nd}f} %".replace(".", ",")


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", t)


def detectar_municipios(pregunta: str, nombres: list[str]) -> list[str]:
    """Municipios mencionados, en el orden en que aparecen. Tolera tildes y nombres incompletos."""
    q = " " + re.sub(r"\s+", " ", _norm(pregunta)) + " "
    hallados = []  # (posición, longitud, nombre)
    for n in nombres:
        nn = " " + _norm(n).strip() + " "
        i = q.find(nn)
        if i >= 0:
            hallados.append((i, len(nn), n))
    # un nombre contenido en otro más largo ya detectado (p. ej. «San Vicente» dentro de «San Vicente de Chucurí») se descarta
    hallados.sort(key=lambda x: -x[1])
    elegidos = []
    for i, ln, n in hallados:
        if not any(i >= j and i + ln <= j + lj for j, lj, _ in elegidos):
            elegidos.append((i, ln, n))
    if not elegidos:  # alias: nombre sin artículo inicial o sus dos primeras palabras, solo si identifican a un único municipio
        alias: dict[str, list[str]] = {}
        for n in nombres:
            pal = _norm(n).split()
            sin_art = [w for k, w in enumerate(pal) if not (k == 0 and w in ("el", "la", "los", "las"))]
            genericas = {"de", "del", "san", "santa", "puerto", "los", "las", "la", "el"}
            palabras_clave = [w for w in sin_art if w not in genericas and len(w) >= 6]
            for a in {" ".join(sin_art), " ".join(sin_art[:2]), *palabras_clave}:
                if len(a.split()) >= 2 or len(a) >= 6:
                    alias.setdefault(a, []).append(n)
        for a, cand in alias.items():
            if len(cand) == 1 and (" " + a + " ") in q and cand[0] not in [e[2] for e in elegidos]:
                elegidos.append((q.find(" " + a + " "), len(a), cand[0]))
    if not elegidos:  # coincidencia aproximada por ventanas de 1 a 3 palabras
        palabras = _norm(pregunta).split()
        mapa = {_norm(n).strip(): n for n in nombres}
        for tam in (3, 2, 1):
            for k in range(len(palabras) - tam + 1):
                ventana = " ".join(palabras[k:k + tam])
                if len(ventana) < 5:
                    continue
                m = difflib.get_close_matches(ventana, list(mapa), n=1, cutoff=0.86)
                if m and mapa[m[0]] not in [e[2] for e in elegidos]:
                    elegidos.append((k, len(ventana), mapa[m[0]]))
    return [n for _, _, n in sorted(elegidos)]


def _numero(pregunta: str, por_defecto: int | None = None) -> int | None:
    m = re.search(r"\b(\d{1,3})\b", pregunta)
    return int(m.group(1)) if m else por_defecto


def _fila(ctx: Contexto, m: str) -> pd.Series:
    return ctx.tabla[ctx.tabla["municipio"] == m].iloc[0]


def _prob(ctx: Contexto, m: str) -> float | None:
    if ctx.proy is None:
        return None
    d = ctx.proy[(ctx.proy["municipio"] == m) & (ctx.proy["anio"] == ctx.anio_fin)]
    return float(d["p_acum_m4"].iloc[0]) if len(d) else None


# ------------------------------------------------------------------ respuestas
def _perfil(ctx: Contexto, m: str) -> dict:
    f = _fila(ctx, m)
    total = ctx.tabla["poblacion"].sum()
    d = ctx.reco[ctx.reco["municipio"] == m]
    decision = (f"**{d.iloc[0]['decision']}**: {d.iloc[0]['accion']}" if len(d)
                else "**Sin acción extraordinaria**: seguimiento de rutina.")
    p = _prob(ctx, m)
    lineas = [
        f"### {m}",
        f"Nivel de riesgo integral **{f['nivel']}** · IRI **{f['iri']:.0f}/100** (puesto {int(f['ranking'])} de {len(ctx.tabla)}).",
        f"- **Exposición:** {miles(f['poblacion'])} habitantes ({pct(f['poblacion'] / total, 1)} de Santander).",
        f"- **Amenaza y terreno:** PGA esperado {dec(f['PGA_medio'], 2)} g · vulnerabilidad del sitio {dec(f['vulnerabilidad'], 2)}.",
        f"- **Actividad en 12 meses:** {int(f['observados_12m'])} sismos M≥4 frente a {dec(f['esperados_12m'])} esperados (semáforo {f['semaforo']}).",
    ]
    if m in ctx.rf:
        lineas.append(f"- **Random Forest:** categoría esperada para {ctx.anio_siguiente}: **{ctx.rf[m]}**.")
    if m in ctx.grupos:
        lineas.append(f"- **K-Means:** pertenece al grupo **{ctx.grupos[m]}** por su perfil sísmico y territorial.")
    if p is not None:
        lineas.append(f"- **Proyección:** probabilidad de al menos un sismo M≥4 entre {ctx.anio_ini} y {ctx.anio_fin}: **{pct(p)}**.")
    if "prioridad_rl" in f.index and pd.notna(f["prioridad_rl"]):
        lineas.append(f"- **Agente de refuerzo:** puesto {int(f['prioridad_rl'])} de {len(ctx.tabla)} para evaluar primero.")
    lineas.append(f"\n**Decisión recomendada →** {decision}")
    return {"texto": "\n".join(lineas), "tabla": None}


def _comparar(ctx: Contexto, a: str, b: str) -> dict:
    fa, fb = _fila(ctx, a), _fila(ctx, b)
    filas = {
        "Índice de Riesgo Integral": (f"{fa['iri']:.0f}/100", f"{fb['iri']:.0f}/100"),
        "Nivel": (fa["nivel"], fb["nivel"]),
        "Habitantes": (miles(fa["poblacion"]), miles(fb["poblacion"])),
        "PGA esperado (g)": (dec(fa["PGA_medio"], 2), dec(fb["PGA_medio"], 2)),
        "Vulnerabilidad (0-1)": (dec(fa["vulnerabilidad"], 2), dec(fb["vulnerabilidad"], 2)),
        "Semáforo 12 meses": (fa["semaforo"], fb["semaforo"]),
        "Categoría RF próximo año": (ctx.rf.get(a, "—"), ctx.rf.get(b, "—")),
    }
    pa, pb = _prob(ctx, a), _prob(ctx, b)
    if pa is not None and pb is not None:
        filas[f"Prob. ≥1 sismo M≥4 hasta {ctx.anio_fin}"] = (pct(pa), pct(pb))
    tabla = pd.DataFrame(filas, index=[a, b]).T.reset_index().rename(columns={"index": "Indicador"})

    mayor, menor = (a, b) if fa["iri"] >= fb["iri"] else (b, a)
    fm, fn = _fila(ctx, mayor), _fila(ctx, menor)
    factores = {"la exposición (la cantidad de población)": (fm["exposicion"], fn["exposicion"]),
                "la amenaza sísmica (PGA)": (fm["amenaza"], fn["amenaza"]),
                "la vulnerabilidad del terreno": (0.5 + 0.5 * fm["vulnerabilidad"], 0.5 + 0.5 * fn["vulnerabilidad"])}
    motivo = max(factores, key=lambda k: abs(math.log(max(factores[k][0], 1e-9) / max(factores[k][1], 1e-9))))
    texto = (f"**{mayor}** es el más prioritario: IRI {dec(fm['iri'], 0)} frente a {dec(fn['iri'], 0)} de **{menor}**. "
             f"La diferencia viene sobre todo de {motivo}. Dicho de otro modo, **{menor}** es el de menor riesgo integral de los dos.")
    if fa["semaforo"] != fb["semaforo"]:
        raro = a if fa["semaforo"] != "Verde" else b
        texto += f" Ojo: **{raro}** muestra actividad inusual en los últimos 12 meses."
    texto += "\n\n*«Mejor» puede significar menos riesgo o más urgencia; aquí se interpreta como prioridad de actuación.*"
    return {"texto": texto, "tabla": tabla}


def _ranking(ctx: Contexto, n: int, menor: bool = False) -> dict:
    t = ctx.tabla.sort_values("iri", ascending=menor).head(n)
    lista = "\n".join(f"{i}. **{r.municipio}** · IRI {r.iri:.0f} · {miles(r.poblacion)} hab. · {r.nivel}" for i, r in enumerate(t.itertuples(), 1))
    if menor:
        encabezado = f"Los {n} municipios con **menor** riesgo integral (índice relativo, no significa que no haya sismos):"
    else:
        encabezado = f"Los {n} municipios con **mayor** riesgo integral:"
    tabla = t[["ranking", "municipio", "iri", "nivel", "poblacion", "semaforo"]].rename(columns={
        "ranking": "Puesto", "municipio": "Municipio", "iri": "IRI", "nivel": "Nivel", "poblacion": "Habitantes", "semaforo": "Semáforo"})
    tabla["Habitantes"] = tabla["Habitantes"].map(miles)
    return {"texto": encabezado + "\n\n" + lista, "tabla": tabla}


def _donde_actuar(ctx: Contexto) -> dict:
    r = ctx.reco
    if r.empty:
        return {"texto": "Hoy no hay municipios que requieran acciones extraordinarias.", "tabla": None}
    p = int(r["prioridad"].min())
    primeros = r[r["prioridad"] == p].head(5)
    nombres = ", ".join(f"**{m}**" for m in primeros["municipio"])
    resto = int((r["prioridad"] == p).sum()) - len(primeros)
    if resto > 0:
        nombres += f" y {resto} más"
    texto = (f"La decisión más urgente hoy es **{ACCIONES[p][0].lower()}** en {nombres}.\n\n"
             f"**Qué hacer:** {ACCIONES[p][1]}\n\n**Por qué:** {primeros.iloc[0]['porque']}\n\n"
             f"Resumen general: {resumen(r)}")
    return {"texto": texto, "tabla": None}


def _alertas(ctx: Contexto) -> dict:
    a = ctx.tabla[ctx.tabla["semaforo"] != "Verde"].sort_values("p_valor")
    if a.empty:
        return {"texto": "Ningún municipio muestra hoy actividad sísmica inusual: todos están en verde.", "tabla": None}
    lineas = "\n".join(f"- **{r.municipio}** ({r.semaforo}): {int(r.observados_12m)} sismos M≥4 en 12 meses frente a {dec(r.esperados_12m)} esperados" for r in a.itertuples())
    return {"texto": "Municipios con actividad inusual en los últimos 12 meses:\n\n" + lineas +
            "\n\n*Es una señal estadística para vigilar, no un pronóstico de un sismo mayor.*", "tabla": None}


def _plan(ctx: Contexto, presupuesto: float) -> dict:
    r = recomendar(ctx.tabla, presupuesto=presupuesto)
    plan = r[r["en_plan"]]
    if plan.empty:
        return {"texto": f"Con {presupuesto:.0f} unidades no alcanza para evaluar ni un municipio (cada evaluación cuesta entre 2 y 5 unidades simuladas).", "tabla": None}
    cubierta = plan["poblacion"].sum()
    tabla = plan[["decision", "municipio", "poblacion", "costo_simulado"]].rename(columns={
        "decision": "Decisión", "municipio": "Municipio", "poblacion": "Habitantes", "costo_simulado": "Costo (simulado)"})
    tabla["Habitantes"] = tabla["Habitantes"].map(miles)
    texto = (f"Con **{presupuesto:.0f} unidades** (costo simulado) se pueden evaluar **{len(plan)} municipios**, que reúnen "
             f"{miles(cubierta)} habitantes ({pct(cubierta / ctx.tabla['poblacion'].sum())} de Santander), en este orden de prioridad:")
    return {"texto": texto, "tabla": tabla}


def _poblacion(ctx: Contexto) -> dict:
    alto = ctx.tabla[ctx.tabla["nivel"] == "Alto"]
    total = ctx.tabla["poblacion"].sum()
    texto = (f"**{miles(alto['poblacion'].sum())} habitantes** ({pct(alto['poblacion'].sum() / total)} de los {miles(total)} de Santander) viven en los "
             f"**{len(alto)} municipios de nivel Alto**: {', '.join(alto['municipio'])}.")
    return {"texto": texto, "tabla": None}


def _proyeccion_top(ctx: Contexto, n: int) -> dict:
    if ctx.proy is None:
        return {"texto": "La proyección no está disponible.", "tabla": None}
    d = ctx.proy[ctx.proy["anio"] == ctx.anio_fin].sort_values("p_acum_m4", ascending=False).head(n)
    lista = "\n".join(f"{i}. **{r.municipio}**: {pct(r.p_acum_m4)}" for i, r in enumerate(d.itertuples(), 1))
    return {"texto": f"Mayor probabilidad de al menos un sismo M≥4 entre {ctx.anio_ini} y {ctx.anio_fin}:\n\n{lista}\n\n"
            "*Los primeros puestos suelen ser municipios del Nido de Bucaramanga, de sismos profundos; por eso la probabilidad no equivale a riesgo.*", "tabla": None}


METODO = {
    "exposicion": "**Exposición** es cuánta gente hay en el camino de la amenaza. Usamos la población DANE 2025 en escala logarítmica "
                  "(`log10(población)/log10(máximo)`), para que las grandes ciudades no eclipsen todo: Bucaramanga vale 1,0 y Los Santos 0,73.",
    "vulnerabilidad": "**Vulnerabilidad** es qué tan mal responde el terreno: 60 % la fracción de suelo blando o rígido (amplifica la sacudida) y 40 % la "
                      "pendiente media (deslizamientos). El tipo de suelo es una estimación a partir de la topografía.",
    "amenaza": "**Amenaza** es qué tan fuerte puede sacudir el suelo. Usamos el PGA esperado a 475 años del modelo oficial de amenaza sísmica del SGC (2020). "
               "Se usa el PGA y no el conteo de epicentros porque casi todos los M≥4 son profundos (Nido de Bucaramanga) y sacuden también a los vecinos.",
    "iri": "**Índice de Riesgo Integral** = Amenaza × Exposición × (0,5 + 0,5 · Vulnerabilidad), normalizado de 0 a 100. Es un índice **relativo** para "
           "priorizar, con pesos propuestos por el equipo; no estima daños ni víctimas.",
    "reglas": "**Reglas de decisión:** *Actuar ya* = nivel Alto con actividad inusual; *Preparar* = nivel Alto con actividad normal; *Vigilar* = actividad "
              "inusual en un municipio Medio o Bajo. El plan cubre municipios por prioridad hasta agotar el presupuesto (costo simulado).",
    "semaforo": "**Semáforo:** compara los sismos M≥4 de los últimos 12 meses con lo esperado según la historia del municipio (2014-2023) con una prueba de "
                "Poisson. Rojo si p < 0,01, amarillo si p < 0,05. Es una señal estadística, no un pronóstico.",
    "proyeccion": "**Proyección:** probabilidad de al menos un sismo M≥4 con P = 1 − e^(−λ·T), donde λ es la tasa anual de cada municipio (2016-2025, suavizada "
                  "hacia el promedio). Es una extrapolación de tasas, no una predicción de eventos.",
    "datos": "**Datos:** catálogo sísmico del SGC (Consulta Experta 1993-2017 y SeisComP 2018-hoy), amenaza sísmica del SGC, población y límites del DANE, "
             "relieve SRTM, fallas GEM y ríos de OpenStreetMap. Lo único simulado es el costo de evaluación del reto de refuerzo.",
}

AYUDA = ("Puedo ayudarte con preguntas como:\n"
         "- *¿Dónde actuar primero?* · *¿Cuáles son los 5 municipios con más riesgo?*\n"
         "- *Compara Bucaramanga con Girón* · *¿Cuál es más riesgoso, Piedecuesta o Floridablanca?*\n"
         "- *¿Qué hago en Villanueva?* · *Cuéntame de Barrancabermeja*\n"
         "- *¿Qué municipios están en alerta?* · *¿Cuánta gente vive en riesgo alto?*\n"
         "- *¿Qué evaluaría con 50 unidades de presupuesto?*\n"
         "- *¿Cómo se calcula el índice?* · *¿Qué es la exposición?*\n\n"
         "Todo lo que respondo sale de los datos del proyecto.")


def _tema_metodo(q: str) -> str | None:
    claves = [("exposicion", r"exposici"), ("vulnerab", r"vulnerab"), ("amenaza", r"amenaza|pga"), ("iri", r"\biri\b|indice|riesgo integral"),
              ("reglas", r"regla|prioridad 1|como se decide|cómo se decide|decision se toma"), ("semaforo", r"semaforo|sem[aá]foro"),
              ("proyeccion", r"proyecci|poisson"), ("datos", r"datos|fuente|variables")]
    for k, patron in claves:
        if re.search(patron, q):
            return "vulnerabilidad" if k == "vulnerab" else k
    return None


def responder(pregunta: str, ctx: Contexto) -> dict:
    """Devuelve {'texto': markdown, 'tabla': DataFrame | None}."""
    q = _norm(pregunta)
    nombres = ctx.tabla["municipio"].tolist()
    munis = detectar_municipios(pregunta, nombres)
    n = _numero(pregunta)

    if not q.strip() or re.search(r"\b(hola|ayuda|que puedes|qué puedes|que sabes)\b", q):
        return {"texto": AYUDA, "tabla": None}
    if len(munis) >= 2:
        return _comparar(ctx, munis[0], munis[1])
    if re.search(r"presupuesto|unidades|plan de|plan\b|evaluar|evaluaria|evaluaría", q) and not munis:
        return _plan(ctx, float(n) if n and n >= 3 else 30.0)
    if re.search(r"alerta|inusual|semaforo|vigilar|actividad reciente|que esta pasando", q) and not munis:
        return _alertas(ctx)
    if re.search(r"cuanta gente|cuantos habitantes|habitantes|poblacion en riesgo|cuantas personas|personas", q) and not munis:
        return _poblacion(ctx)
    if re.search(r"probabilidad|proyecci|2027|futuro|proximo ano|proximos anos", q) and not munis:
        return _proyeccion_top(ctx, n if n and n <= 15 else 5)
    if len(munis) == 1:
        return _perfil(ctx, munis[0])
    tema = _tema_metodo(q)
    if re.search(r"como se calcula|como funciona|que es|que significa|explica|formula|por que|cual es la formula", q) and tema:
        return {"texto": METODO[tema], "tabla": None}
    if re.search(r"menos riesgo|menor riesgo|mas seguro|mas segura|menos peligro|mas tranquilo", q):
        return _ranking(ctx, n if n and n <= 15 else 5, menor=True)
    if re.search(r"donde actuar|primero|urgente|que hago|que hacer|decision|recomienda|prioridad", q) and not re.search(r"cuales|top|ranking", q):
        return _donde_actuar(ctx)
    if re.search(r"mas riesgo|mayor riesgo|mas peligro|top|ranking|cuales|mas riesgoso|critico|críticos", q):
        return _ranking(ctx, n if n and n <= 15 else 5)
    if tema:
        return {"texto": METODO[tema], "tabla": None}
    return {"texto": "No estoy seguro de haber entendido la pregunta. " + AYUDA, "tabla": None}
