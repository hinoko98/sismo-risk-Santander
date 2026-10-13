"""Ficha municipal de riesgo sísmico (HTML autónomo, lista para imprimir o guardar como PDF)."""
import base64
from html import escape

from .config import RUTA_LOGO

COLORES = {"Alto": "#e74c3c", "Medio": "#f1c40f", "Bajo": "#2ecc71", "Rojo": "#e74c3c", "Amarillo": "#f1c40f", "Verde": "#2ecc71"}


def _logo_base64() -> str:
    try:
        return "data:image/png;base64," + base64.b64encode(RUTA_LOGO.read_bytes()).decode()
    except Exception:
        return ""


def _chip(texto: str) -> str:
    return f"<span class='chip' style='background:{COLORES.get(texto, '#888')}'>{escape(str(texto))}</span>"


def generar_ficha(d: dict) -> str:
    """`d` reúne los datos ya calculados del municipio (ver app.py). Devuelve el HTML completo."""
    filas = "".join(f"<tr><td>{escape(k)}</td><td><b>{v}</b></td></tr>" for k, v in d["indicadores"].items())
    return f"""<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Ficha de riesgo sísmico · {escape(d['municipio'])}</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;color:#14263d;max-width:820px;margin:24px auto;padding:0 18px}}
 h1{{color:#1B3A5C;margin:6px 0}} h2{{color:#1B3A5C;border-bottom:2px solid #E8601C;padding-bottom:4px;margin-top:26px}}
 .chip{{color:#fff;padding:3px 12px;border-radius:12px;font-weight:600}} table{{border-collapse:collapse;width:100%}}
 td{{border-bottom:1px solid #e3e8ef;padding:6px 8px}} td:first-child{{color:#4b5d73;width:60%}}
 .aviso{{background:#fff6ee;border-left:4px solid #E8601C;padding:10px 14px;font-size:13px;margin-top:26px}}
 .pie{{font-size:12px;color:#6b7b8f;margin-top:18px}}
</style></head><body>
<img src='{_logo_base64()}' alt='Sismo-Risk Santander' style='height:90px'>
<h1>{escape(d['municipio'])}</h1>
<p>Índice de Riesgo Integral: <b>{d['iri']:.1f}/100</b> · nivel {_chip(d['nivel'])} · puesto <b>{d['ranking']} de {d['total']}</b> en Santander</p>
<h2>Indicadores</h2><table>{filas}</table>
<h2>Decisión recomendada</h2>
<p><b>{escape(d['decision'])}</b></p>
<p style='color:#4b5d73;font-size:14px'>{escape(d['porque'])}</p>
<h2>Actividad sísmica reciente</h2>
<p>Últimos 12 meses (catálogo SGC, M≥4): <b>{d['obs12']}</b> observados frente a <b>{d['esp12']}</b> esperados &nbsp;{_chip(d['semaforo'])}</p>
<h2>Proyección</h2>
<p>Probabilidad de al menos un sismo M≥4 en el municipio entre {d['anio_ini']} y {d['anio_fin']}: <b>{d['p2027']:.0%}</b>.<br>
Categoría esperada para el próximo año (modelo Random Forest): <b>{escape(d['clase_rf'])}</b>.</p>
<div class='aviso'><b>Cómo leer esta ficha.</b> El índice es relativo y sirve para priorizar; no estima daños ni víctimas. Un sismo
concreto no se puede predecir: las probabilidades describen la tasa histórica. El tipo de suelo es una estimación a partir de la pendiente.</div>
<p class='pie'>Fuentes: SGC (catálogo sísmico y amenaza sísmica 2020), DANE (población 2025 y límites), GEM (fallas),
SRTM (relieve). Generado por Sismo-Risk Santander · Hackathon CIAI UTSmart 2026.</p>
</body></html>"""
