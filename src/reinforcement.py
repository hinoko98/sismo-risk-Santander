"""Reto de refuerzo: priorización de evaluaciones con presupuesto limitado (Q-learning tabular).

Problema: cada año se dispone de un presupuesto para enviar equipos técnicos de evaluación a algunos
municipios. El agente ve los municipios en secuencia y decide EVALUAR o NO EVALUAR.

  * Estado     : actividad reciente (M≥3 de los 3 años previos), población expuesta, costo, presupuesto restante,
                 amenaza estimada (PGA del SGC) y grupo del agrupamiento no supervisado (resultado del reto medio).
  * Recompensa : beneficio – costo.  El beneficio usa resultados REALES del año siguiente
                 (la categoría que el municipio tuvo) y la población real del DANE.
  * Costo      : SIMULADO (el reto lo permite): sube con el área del municipio y su lejanía de Bucaramanga.
                 No son valores monetarios reales, solo unidades relativas de esfuerzo.

El entrenamiento usa los años más antiguos y la evaluación los más recientes (sin mirar el futuro).
"""
import numpy as np
import pandas as pd

from .config import OBJETIVO, SEMILLA

PRESUPUESTO = 30.0                       # unidades de costo disponibles por año
SEVERIDAD = {1: 0.0, 2: 0.4, 3: 1.0}     # cuánto vale haber evaluado un municipio según su categoría real
ESCALA_BENEFICIO = 1000.0  # parámetro simulado: valor relativo de cubrir a toda la población con riesgo Alto
ALPHA, GAMMA = 0.10, 0.95


def costo_simulado(area_km2, dist_bucaramanga_km):
    """Costo SIMULADO de evaluar un municipio (unidades relativas)."""
    return 2.0 + area_km2 / 500.0 + dist_bucaramanga_km / 50.0


class EntornoEvaluacion:
    """Un episodio = un año; los municipios llegan en orden aleatorio."""

    def __init__(self, panel: pd.DataFrame, referencia: pd.DataFrame, grupos: dict):
        self.panel = panel
        pob_total = panel.drop_duplicates("municipio")["poblacion"].sum()
        panel = panel.assign(
            costo=costo_simulado(panel["area_km2"], panel["distancia_bucaramanga_km"]),
            peso_pob=panel["poblacion"] / pob_total,
        )
        # Los cortes de los terciles se fijan con el periodo de entrenamiento (referencia)
        r = referencia.assign(costo=costo_simulado(referencia["area_km2"], referencia["distancia_bucaramanga_km"]))
        self.cortes = {c: r[c].quantile([1 / 3, 2 / 3]).to_numpy()
                       for c in ["n_m3_prev3", "poblacion", "costo", "PGA_medio"]}
        self.grupos = grupos  # municipio -> 0 (Bajo) · 1 (Medio) · 2 (Alto), del K-Means
        self.panel = panel

    def _nivel(self, col, valor):
        return int(np.digitize(valor, self.cortes[col]))

    @staticmethod
    def nivel_presupuesto(restante):
        return 0 if restante < 0.25 * PRESUPUESTO else (1 if restante < 0.6 * PRESUPUESTO else 2)

    def estado(self, fila, restante):
        return (self._nivel("n_m3_prev3", fila["n_m3_prev3"]), self._nivel("poblacion", fila["poblacion"]),
                self._nivel("costo", fila["costo"]), self.nivel_presupuesto(restante),
                self.grupos.get(fila["municipio"], 1), self._nivel("PGA_medio", fila["PGA_medio"]))

    @staticmethod
    def beneficio(fila):
        return ESCALA_BENEFICIO * fila["peso_pob"] * SEVERIDAD[int(fila[OBJETIVO])]


def _episodio(entorno, anio, rng, politica, aprender=None):
    """Recorre los municipios de un año aplicando `politica(estado, fila, restante) -> 0/1`."""
    filas = entorno.panel[entorno.panel["anio"] == anio].sample(frac=1, random_state=int(rng.integers(1e9)))
    restante, total, ben, usado, cubiertos_alto, n_alto = PRESUPUESTO, 0.0, 0.0, 0.0, 0, 0
    registros = list(filas.to_dict("records"))
    for i, f in enumerate(registros):
        s = entorno.estado(f, restante)
        puede = f["costo"] <= restante
        a = politica(s, f, restante) if puede else 0
        r = 0.0
        if a == 1:
            r = entorno.beneficio(f) - f["costo"]
            restante -= f["costo"]
            usado += f["costo"]
            ben += entorno.beneficio(f)
        n_alto += int(f[OBJETIVO] == 3)
        cubiertos_alto += int(a == 1 and f[OBJETIVO] == 3)
        total += r
        if aprender is not None:
            sig = entorno.estado(registros[i + 1], restante) if i + 1 < len(registros) else None
            aprender(s, a, r, sig)
    return {"recompensa": total, "beneficio": ben, "costo": usado, "altos_cubiertos": cubiertos_alto, "altos": n_alto}


def entrenar_agente(panel: pd.DataFrame, anios_train: list[int], grupos: dict, epocas: int = 400, seed: int = SEMILLA):
    """Q-learning ε-greedy sobre los años de entrenamiento. Devuelve (Q, entorno, curva de aprendizaje)."""
    rng = np.random.default_rng(seed)
    datos = panel.dropna(subset=[OBJETIVO])
    referencia = datos[datos["anio"].isin(anios_train)]
    entorno = EntornoEvaluacion(datos, referencia, grupos)
    Q = np.zeros((3, 3, 3, 3, 3, 3, 2))
    curva = []
    for ep in range(epocas):
        eps = max(0.05, 1.0 - ep / (0.7 * epocas))

        def politica(s, f, restante):
            return int(rng.integers(2)) if rng.random() < eps else int(Q[s][1] > Q[s][0])

        def aprender(s, a, r, sig):
            objetivo = r + (GAMMA * Q[sig].max() if sig is not None else 0.0)
            Q[s][a] += ALPHA * (objetivo - Q[s][a])

        res = [_episodio(entorno, anio, rng, politica, aprender)["recompensa"] for anio in rng.permutation(anios_train)]
        curva.append(float(np.mean(res)))
    return Q, entorno, curva


def evaluar_politicas(Q, entorno, anios_test: list[int], repeticiones: int = 30, seed: int = SEMILLA) -> pd.DataFrame:
    """Compara el agente con estrategias de referencia en los años de prueba (promedio de varias llegadas aleatorias)."""
    rng = np.random.default_rng(seed)
    politicas = {
        "Agente RL (Q-learning)": lambda s, f, r: int(Q[s][1] > Q[s][0]),
        "Aleatoria (30 %)": lambda s, f, r: int(rng.random() < 0.30),
        "Regla: actividad alta y población ≥ media": lambda s, f, r: int(s[0] == 2 and s[1] >= 1),
        "Evaluar todo mientras alcance": lambda s, f, r: 1,
    }
    filas = []
    for nombre, pol in politicas.items():
        res = [_episodio(entorno, a, rng, pol) for a in anios_test for _ in range(repeticiones)]
        d = pd.DataFrame(res)
        filas.append({
            "Política": nombre,
            "Recompensa neta media/año": d["recompensa"].mean(),
            "Beneficio medio/año": d["beneficio"].mean(),
            "Costo medio/año": d["costo"].mean(),
            "% municipios Alto evaluados": 100 * d["altos_cubiertos"].sum() / max(d["altos"].sum(), 1),
        })
    return pd.DataFrame(filas).round(2)


def prioridades(Q, entorno, panel_actual: pd.DataFrame) -> pd.DataFrame:
    """Ranking de municipios para el año que viene según la ventaja aprendida de evaluar (presupuesto completo)."""
    p = panel_actual.assign(
        costo=costo_simulado(panel_actual["area_km2"], panel_actual["distancia_bucaramanga_km"]))
    filas = []
    for f in p.to_dict("records"):
        s = entorno.estado(f, PRESUPUESTO)
        filas.append({"municipio": f["municipio"], "ventaja_evaluar": Q[s][1] - Q[s][0], "costo_simulado": f["costo"],
                      "poblacion": f["poblacion"], "sismos_M3_prom_3a": f["n_m3_prev3"]})
    return pd.DataFrame(filas).sort_values("ventaja_evaluar", ascending=False).reset_index(drop=True)


def grupos_no_supervisados(anio_max: int) -> dict:
    """Grupo K-Means (0 Bajo · 1 Medio · 2 Alto) de cada municipio usando SOLO sismicidad hasta `anio_max`,
    para que el agente no vea información de los años de prueba."""
    from .config import RUTA_MUN_FEATURES, RUTA_SISMOS_LIMPIO
    from .unsupervised import agregar_municipios, agrupar
    agg = agregar_municipios(pd.read_csv(RUTA_SISMOS_LIMPIO), pd.read_csv(RUTA_MUN_FEATURES), anio_max=anio_max)
    tabla, _ = agrupar(agg, 3)
    return {m: {"Bajo": 0, "Medio": 1, "Alto": 2}[g] for m, g in zip(tabla["municipio"], tabla["grupo"])}


def entrenar_y_guardar(panel: pd.DataFrame, guardar: bool = True, epocas: int = 300) -> dict:
    """Entrena con los años antiguos, evalúa con los recientes y guarda el agente en models/."""
    import joblib
    from .config import RUTA_AGENTE_RL
    from .model_training import dividir_temporal

    datos = panel.dropna(subset=[OBJETIVO])
    train, test = dividir_temporal(datos)
    anios_train, anios_test = sorted(train["anio"].unique().tolist()), sorted(test["anio"].unique().tolist())
    grupos = grupos_no_supervisados(max(anios_train))
    Q, entorno, curva = entrenar_agente(panel, anios_train, grupos, epocas=epocas)
    resultado = {
        "Q": Q, "curva": curva, "anios_train": anios_train, "anios_test": anios_test,
        "comparacion": evaluar_politicas(Q, entorno, anios_test),
        "prioridades": None,
    }
    from .config import ANIO_ACTUAL
    resultado["prioridades"] = prioridades(Q, entorno, panel[panel["anio"] == ANIO_ACTUAL])
    if guardar:
        RUTA_AGENTE_RL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(resultado, RUTA_AGENTE_RL)
    return resultado


if __name__ == "__main__":
    from .config import RUTA_PANEL
    r = entrenar_y_guardar(pd.read_csv(RUTA_PANEL))
    print(r["comparacion"].to_string())
    print(r["prioridades"].head(8).to_string())
