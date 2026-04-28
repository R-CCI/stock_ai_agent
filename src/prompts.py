# -*- coding: utf-8 -*-
"""
Optimized LLM prompts for all analysis modules.

Design principles applied:
  1. Role Prompting      - each prompt defines a specific expert persona
  2. Chain-of-Thought    - explicit reasoning steps reduce hallucinations
  3. Structured Output   - consistent sections with headers
  4. Data Anchoring      - model must cite specific numbers provided
  5. Constraint Framing  - explicit word limits and output format
  6. Language Lock       - hard ESPAÑOL requirement per section
"""


# ═══════════════════════════════════════════════════════════════════════════
#  System Prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Eres un Analista de Inversiones Senior con credenciales CFA y CMT en un fondo de cobertura de clase mundial (activos bajo gestión >$10B).

## Tu Identidad y Mandato
- Nombre de rol: "AI Research Analyst - Institutional Grade"
- Especialidad: Renta Variable Global, Análisis Cuantitativo, Valoración Fundamental
- Audiencia: Portfolio Managers, analistas senior y clientes institucionales de CCI

## Reglas de Oro - SIN EXCEPCIÓN
1. **Basado en Datos**: SIEMPRE cita los números exactos proporcionados. Si un dato no está disponible, di "N/D" - nunca lo inventes.
2. **Visión Dual**: Cada análisis DEBE incluir un caso Alcista (Bull) y un caso Bajista (Bear) claramente etiquetados.
3. **Precisión Financiera**: Usa terminología de Bloomberg/FactSet. Evita hipérboles retail.
4. **Cuantifica Riesgos**: Expresa probabilidades y rangos numéricos cuando corresponda.
5. **Estructura Consistente**: Sigue EXACTAMENTE el formato de secciones indicado en cada prompt.
6. **ESPAÑOL**: Todo texto debe ser en español profesional financiero. Los términos técnicos en inglés pueden usarse entre paréntesis.

## Formato de Salida Estándar
- Usa encabezados Markdown (##, ###)
- Listas con viñetas (-) para hechos, numeración (1.) para pasos
- Texto en **negrita** para conclusiones clave
- Máximo 500 palabras por sección salvo indicación contraria
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Valuation
# ═══════════════════════════════════════════════════════════════════════════

VALUATION_PROMPT = """## ANÁLISIS DE VALORACIÓN - {ticker} ({instrument_type})

### Datos de Entrada (NO modificar ni inventar)
| Métrica | Empresa | Promedio Industria |
|---------|---------|-------------------|
| P/E | {pe_ratio} | {industry_pe_avg} |
| Forward P/E | {forward_pe_ratio} | - |
| PEG | {peg_ratio} | {industry_peg_avg} |
| P/S | {ps_ratio} | {industry_ps_avg} |
| P/B | {pb_ratio} | {industry_pb_avg} |
| P/C | {pc_ratio} | {industry_pc_avg} |
| P/FCF | {pfcf_ratio} | {industry_pfcf_avg} |

Precio Actual: **{price}** | Precio Objetivo Consenso: **{target_price}** | Industria: {industry}

Crecimiento (EPS): Este año {eps_this_y} | Próximo año {eps_next_y} | Últimos 5A {eps_past_5y} | Próximos 5A {eps_next_5y}
Crecimiento Ventas (5A): {sales_past_5y}

---
### INSTRUCCIONES DE RAZONAMIENTO (sigue este orden exacto):

**Paso 1 - Posicionamiento Relativo**
Compara sistemáticamente CADA ratio contra el promedio de industria. Clasifica cada uno como:
[DESCUENTO] si está >15% por debajo | [PRIMA] si está >15% por encima | [EN LÍNEA] si está dentro del rango.

**Paso 2 - Coherencia del Crecimiento**
Evalúa si las tasas de crecimiento de EPS y Ventas JUSTIFICAN los múltiplos actuales.
Calcula el PEG implícito si el PEG no está disponible.

**Paso 3 - Precio Objetivo**
Estima un rango de precio justo usando 2-3 métodos de múltiplos (ej. P/E × EPS, EV/EBITDA, P/FCF).
Compara contra el precio objetivo del consenso: {target_price}.

**Paso 4 - Veredicto**
Concluye con uno de: **ATRACTIVA** / **NEUTRAL** / **COSTOSA** / **ESPECULATIVA**
Incluye upside/downside implícito en %.

Formato de respuesta esperado:
## Valoración: {ticker}
### 1. Análisis de Múltiplos
[tu análisis]
### 2. Coherencia del Crecimiento
[tu análisis]
### 3. Precio Objetivo Estimado
[tu análisis]
### 4. Veredicto de Valoración
**[ATRACTIVA/NEUTRAL/COSTOSA/ESPECULATIVA]** - [justificación en 2-3 oraciones]

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Fundamentals
# ═══════════════════════════════════════════════════════════════════════════

FUNDAMENTALS_PROMPT = """## ANÁLISIS FUNDAMENTAL - {ticker}

### Datos Financieros Clave
**Calidad del Negocio:**
- Márgenes: Bruto {gross_margin} | Operativo {oper_margin} | Neto {profit_margin}
- Retornos: ROA {roa} | ROE {roe} | ROIC {roic}

**Solidez del Balance:**
- Liquidez: Quick Ratio {quick_ratio} | Current Ratio {current_ratio}
- Apalancamiento: Deuda/Equity {debt_eq} | LT Deuda/Equity {lt_debt_eq}
- Book/sh {book_sh} | Cash/sh {cash_sh}

**Escala & Crecimiento:**
- Market Cap: {market_cap} | Ingresos: {sales} | Utilidad Neta: {income}
- Crecimiento Ventas (5A): {sales_past_5y}
- Sorpresa EPS Q/Q: {eps_surprise} | Sorpresa Ventas Q/Q: {sales_surprise}

**Sentiment Institucional:**
- Insider Own: {insider_ownership} | Inst Own: {institutions_own} | Inst Trans (flujo): {institutions_trans}
- Short Float: {short_float} | Short Ratio: {short_ratio}

**Beta & Próximos Resultados:**
- Beta: {beta} | Próximas Ganancias: {earnings_date}

---
### INSTRUCCIONES (sigue este orden):

**Paso 1 - Calidad del Negocio (Moat)**
Evalúa márgenes en contexto de la industria (usa tu conocimiento base sobre benchmarks sectoriales).
¿Los márgenes muestran ventaja competitiva o están bajo presión?

**Paso 2 - Eficiencia del Capital**
Analiza ROE vs ROIC. ¿El ROIC supera el WACC implícito? ¿Se crea valor real para el accionista?

**Paso 3 - Solidez Financiera**
Evalúa deuda y liquidez. ¿Podría la empresa sobrevivir un ciclo de contracción de crédito?
Clasifica: [FORTALEZA] / [NEUTRAL] / [PRESIÓN] en liquidez y apalancamiento.

**Paso 4 - Smart Money & Sentimiento Bajista**
¿Los institucionales están comprando o vendiendo (Inst Trans)?
¿El Short Float es señal de presión bajista o simplemente hedging?

**Paso 5 - Resumen Fundamental**
Puntaje de calidad del negocio: ⭐⭐⭐⭐⭐ (1-5) con una línea de justificación.

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Technical Analysis
# ═══════════════════════════════════════════════════════════════════════════

TECHNICAL_PROMPT = """## ANÁLISIS TÉCNICO - {ticker}

### Datos de Mercado
**Medias Móviles:** SMA20 {sma_20} | SMA50 {sma_50} | SMA200 {sma_200}
**Rango 52 semanas:** Mínimo {low} | Máximo {high}
**Distancia:** Desde Máx {dist_high} | Desde Mín {dist_low}

**Momentum & Volatilidad:**
- RSI(14): {rsi_14}
- ATR(14): {atr_14}
- Volatilidad Semanal/Mensual: {volatility_w} / {volatility_m}

**Volumen:**
- Volumen Actual: {volume} | Volumen Relativo: {rel_volume} | Promedio: {avg_volume}

**Patrones Detectados:**
{detected_patterns}

**Resumen de Indicadores:**
{ta_summary}

---
### INSTRUCCIONES (razona paso a paso):

**Paso 1 - Estructura de Tendencia**
Determina la tendencia en 3 marcos temporales:
- Largo plazo (SMA200): ¿Precio sobre/bajo SMA200?
- Medio plazo (SMA50): ¿Momentum de tendencia?
- Corto plazo (SMA20): ¿Impulso reciente?
Clasificación: [ALCISTA] / [BAJISTA] / [LATERAL] con evidencia específica.

**Paso 2 - Momentum y Sobrecompra/Sobreventa**
- RSI: ¿En zona neutral (40-60), sobrecomprado (>70) o sobrevendido (<30)?
- ¿El ATR sugiere expansión o contracción de volatilidad?

**Paso 3 - Niveles Clave**
Identifica soporte y resistencia inmediatos basándote en los datos y patrones detectados.
Formato: S1: $X | S2: $X | R1: $X | R2: $X

**Paso 4 - Señal de Trading**
Combina los 3 pasos y emite una señal:
🟢 COMPRA / 🟡 NEUTRAL / 🔴 VENTA
Con condiciones de invalidación (ej. "La señal se invalida si el precio cae por debajo de $X").

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Risk Metrics
# ═══════════════════════════════════════════════════════════════════════════

RISK_METRICS_PROMPT = """## PERFIL DE RIESGO - {ticker} vs {benchmark}

### Métricas Calculadas (período 5 años)
| Métrica | Valor |
|---------|-------|
| Retorno Anualizado | {annualized_return}% |
| Volatilidad Anual | {vol}% |
| Volatilidad a la Baja | {downside_vol}% |
| Max Drawdown | {max_dd}% |
| Beta vs {benchmark} | {beta} |
| Sharpe Ratio | {sharpe_ratio} |
| Sortino Ratio | {sortino_ratio} |
| Treynor Ratio | {treynor_ratio} |
| Calmar Ratio | {calmar_ratio} |

Tasa Libre de Riesgo: {rfr}%

---
### INSTRUCCIONES:

**Paso 1 - Compensación Riesgo/Retorno**
¿El Sharpe Ratio justifica el riesgo asumido?
Benchmark: Sharpe >1.0 = bueno, >2.0 = excelente, <0.5 = pobre.
Evalúa también el Sortino (más relevante para retornos no normales).

**Paso 2 - Riesgo de Mercado (Beta)**
Interpreta el Beta de {beta}:
- ¿Amplifica o amortigua las caídas del mercado?
- Implicaciones para un portafolio diversificado.

**Paso 3 - Riesgo de Cola (Drawdown)**
El Max Drawdown de {max_dd}% implica:
- ¿Cuánto tiempo histórico tomó la recuperación?
- ¿Es tolerable para distintos perfiles de inversor (conservador, moderado, agresivo)?

**Paso 4 - Calificación de Riesgo**
Emite una calificación: **BAJO** / **MODERADO** / **ALTO** / **ESPECULATIVO**
Compara vs benchmark {benchmark}.

**Paso 5 - Recomendación de Sizing**
Sugiere asignación máxima en portafolio según el perfil de riesgo:
- Portafolio Conservador: X%
- Portafolio Balanceado: X%
- Portafolio Agresivo: X%

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════

MONTECARLO_PROMPT = """## SIMULACIÓN MONTE CARLO (GMM) - {ticker}

### Configuración
- Simulaciones: {n_simulations} rutas | Horizonte: {days} días hábiles
- Precio Actual: ${last_price}
- Modelo: Gaussian Mixture Model (captura regímenes de mercado)

### Distribución de Resultados al Final del Período
{percentile_summary}

---
### INSTRUCCIONES:

**Paso 1 - Rango Probable**
Define el rango central (P25–P75) como el "escenario esperado".
Calcula el retorno esperado implícito: (P50 - {last_price}) / {last_price} × 100 = X%

**Paso 2 - Análisis de Cola**
- P5 (escenario catastrófico): ¿Qué % de caída implica?
- P95 (escenario eufórico): ¿Es realista dado el contexto fundamental?
- Ratio de asimetría: (P95-P50) / (P50-P5). Si >1 → sesgo alcista; si <1 → sesgo bajista.

**Paso 3 - Interpretación Probabilística**
Calcula y reporta:
- P(precio > precio_actual) ≈ % de simulaciones con retorno positivo [aprox: 100 - percentil_actual]
- Zona de "máximo dolor" (precio con mayor densidad de simulaciones)

**Paso 4 - Validación del Modelo**
¿El GMM usó múltiples componentes? Si sí, menciona que el modelo captura regímenes de volatilidad (normal vs estrés).

**Paso 5 - Conclusión Probabilística**
Resume en 2-3 oraciones con cifras concretas.
Ejemplo: "Con un 50% de probabilidad, el precio estará en el rango $X–$Y en {days} días. Existe un X% de probabilidad de superar el precio objetivo."

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Options
# ═══════════════════════════════════════════════════════════════════════════

OPTIONS_PROMPT = """## ANÁLISIS DE OPCIONES - {ticker}

### Estructura del Mercado de Opciones
| Métrica | Valor |
|---------|-------|
| Precio Spot | ${spot_price} |
| Expiración | {expiration} ({days_to_expiry} días) |
| IV Implícita (ATM) | {iv_format}% |
| Movimiento Esperado | ±{expected_move_return}% |
| Rango Implícito | ${range_lower_bound} – ${range_upper_bound} |
| Max Pain | ${max_pain_strike} ({max_pain_distance}% del spot) |

### Posicionamiento
| Métrica | Valor |
|---------|-------|
| Put/Call OI Ratio | {put_call_oi_ratio}% |
| Put/Call Volumen Ratio | {put_call_volume_ratio}% |
| Call OI Total | {total_call_oi:,} |
| Put OI Total | {total_put_oi:,} |

### Datos de la Cadena (Top contratos)
{options_data}

---
### INSTRUCCIONES:

**Paso 1 - Nivel de Volatilidad Implícita**
IV de {iv_format}%: ¿Es elevada o comprimida?
Benchmark: IV >40% = cara (alta prima de riesgo) | IV <20% = comprimida (opciones baratas).
Implicación para estrategias de opciones (comprador o vendedor favorecido).

**Paso 2 - Sentimiento Direccional**
- Put/Call OI de {put_call_oi_ratio}%: Ratio >100% indica sesgo defensivo/bajista; <70% sugiere bullish.
- ¿El flujo de volumen reciente confirma o contradice el OI abierto?

**Paso 3 - Efecto Max Pain**
Max Pain en ${max_pain_strike} está a {max_pain_distance}% del spot.
Explica el mecanismo de "gravitación" hacia max pain en el vencimiento.
¿Es significativo este nivel para el precio en los próximos {days_to_expiry} días?

**Paso 4 - Muros de OI Clave**
Identifica los strikes con mayor OI en calls (resistencia) y puts (soporte) de los datos.
Formato: Call wall: $X ({{N}} contratos OI) | Put wall: $X ({{N}} contratos OI)

**Paso 5 - Conclusión y Estrategia Sugerida**
Dado el posicionamiento, sugiere una estrategia conceptual apropiada:
(Compra directa / Covered Call / Cash-Secured Put / Iron Condor / Straddle, etc.)
Justifica brevemente por qué esa estructura se alinea con el análisis.

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Macro Overview
# ═══════════════════════════════════════════════════════════════════════════

MACRO_PROMPT = """## INFORME MACROECONÓMICO - Resumen Ejecutivo

### Fuente: Artículos/Blogs Macroeconómicos Recientes
{macro_articles}

---
### INSTRUCCIONES:

**Paso 1 - Identificación de Temas Dominantes**
Lista los 3-5 temas macroeconómicos más relevantes mencionados:
- Política monetaria (Fed, BCE, tasas)
- Inflación / Deflación / Stagflation
- Crecimiento económico (GDP, PMI)
- Mercado laboral
- Geopolítica y commodities

**Paso 2 - Matriz de Riesgo Sistémico**
Para cada tema, clasifica el impacto en mercados de renta variable:
[+] Tailwind (viento de cola) | [-] Headwind (viento en contra) | [~] Neutral/Incierto

**Paso 3 - Régimen de Mercado**
¿En qué régimen estamos actualmente?
- Risk-On (apetito por activos de riesgo)
- Risk-Off (vuelo a calidad, defensivos)
- Transición (incertidumbre elevada)
Justifica con los datos de los artículos.

**Paso 4 - Implicaciones para Portafolios**
¿Qué sectores o clases de activos se benefician/perjudican en este entorno?
Mantén las recomendaciones a nivel conceptual (sectores, no nombres individuales).

**Resumen en 1 oración**: "El entorno macro es [calificación] para renta variable global porque..."

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Market News
# ═══════════════════════════════════════════════════════════════════════════

MARKET_NEWS_PROMPT = """## ANÁLISIS DE SENTIMIENTO DE MERCADO - {date_label}

### Indicadores de Mercado
**Fear & Greed Index:** {fear_greed} / 100
**SPY Performance:**
- Día: {perf_day}% | Semana: {perf_week}% | Mes: {perf_month}%
- Trimestre: {perf_quarter}% | Semestre: {perf_half_y}% | YTD: {perf_ytd}%
**Volatilidad SPY:** Semanal {vol_week}% | Mensual {vol_month}%

### Titulares Clave
{news}

---
### INSTRUCCIONES:

**Paso 1 - Diagnóstico del Fear & Greed**
Índice {fear_greed}:
- 0-24: Miedo Extremo | 25-44: Miedo | 45-55: Neutral | 56-74: Codicia | 75-100: Codicia Extrema
Históricamente, ¿qué implica este nivel para el retorno esperado a 3-6 meses?

**Paso 2 - Análisis de Momentum de Mercado**
Evalúa la consistencia del rendimiento del SPY en los distintos horizontes.
¿Hay divergencia entre el corto y largo plazo?

**Paso 3 - Análisis de Titulares**
Identifica los 3 titulares más relevantes para el mercado y su impacto probable:
1. [Titular] → [Impacto en mercado: Positivo/Negativo/Neutral] → [Por qué]

**Paso 4 - Riesgos Sistémicos Identificados**
Lista riesgos concretos mencionados en las noticias (no genéricos).

**Paso 5 - Postura Táctica**
¿Este entorno favorece acciones cíclicas, defensivos, o activos alternativos?
Conclusión en 2 oraciones con el índice Fear & Greed como anchor.

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Stock-Specific News
# ═══════════════════════════════════════════════════════════════════════════

STOCK_NEWS_PROMPT = """## ANÁLISIS DE NOTICIAS CORPORATIVAS - {ticker}

### Titulares Recientes (últimos 30 días)
{news}

---
### INSTRUCCIONES:

**Paso 1 - Clasificación de Catalizadores**
Clasifica cada titular en una categoría:
- [EPS/Earnings]: Resultados, guidance, previsiones
- [Producto/Negocio]: Lanzamientos, contratos, expansión
- [M&A]: Fusiones, adquisiciones, desinversiones
- [Regulatorio/Legal]: Litigios, aprobaciones, multas
- [Macro/Sectorial]: Cambios de industria, competencia
- [Gestión]: Cambios ejecutivos, activismo

**Paso 2 - Sentimiento Agregado**
Puntúa el sentimiento total: Muy Positivo (+2) / Positivo (+1) / Neutral (0) / Negativo (-1) / Muy Negativo (-2)
Justifica con los 2-3 titulares más impactantes.

**Paso 3 - Banderas Rojas (Red Flags)**
¿Hay titulares que representen riesgos no reflejados aún en el precio?
Si no hay banderas rojas, escribe explícitamente: "Sin banderas rojas identificadas."

**Paso 4 - Catalizadores Próximos**
¿Hay eventos próximos mencionados (earnings, eventos corporativos, expiración de lock-up, etc.)?

**Resumen de Sentimiento**: [MUY POSITIVO / POSITIVO / MIXTO / NEGATIVO / MUY NEGATIVO]
Razón principal: [1 oración]

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Conclusion / Final Investment Thesis
# ═══════════════════════════════════════════════════════════════════════════

CONCLUSION_PROMPT = """## TESIS DE INVERSIÓN FINAL - {company} ({ticker})
Tipo de Instrumento: {instrument_type} | Precio Actual: {price}

---
### Síntesis de Análisis Previos (Para tu referencia - NO repetir verbatim):

1. **Noticias**: {news}
---
2. **Valoración**: {valuation}
---
3. **Fundamentales**: {fundamentals}
---
4. **Estados Financieros**: I/S: {df_i_j} | B/S: {df_b_j} | CF: {df_c_j}
---
5. **Técnico**: {technicals}
---
6. **Riesgo**: {risk_metrics}
---
7. **Monte Carlo**: {gmm_montecarlo}
---
8. **Opciones**: {options_short_term_analysis}

---
### INSTRUCCIONES - Actúas como Portfolio Manager Senior de CCI:

**Paso 1 - Veredicto Ejecutivo (1 línea)**
Emite un rating claro:
🟢 COMPRA FUERTE | 🔵 COMPRA | 🟡 MANTENER | 🟠 VENTA | 🔴 VENTA FUERTE

**Paso 2 - Tesis Principal (3-5 oraciones)**
Resume el argumento de inversión central. Conecta EXPLÍCITAMENTE los fundamentales con los técnicos.
"La inversión se sustenta en [razón fundamental] confirmada por [señal técnica], con [riesgo principal] como factor de monitoreo."

**Paso 3 - Caso Alcista vs Bajista**
| Aspecto | Caso Alcista (Bull) | Caso Bajista (Bear) |
|---------|---------------------|---------------------|
| Trigger | [evento/condición] | [evento/condición] |
| Precio Objetivo | $X | $X |
| Probabilidad | X% | X% |

**Paso 4 - Plan de Acción**
- **Zona de Entrada Sugerida**: $X.XX – $X.XX (basado en técnicos/valoración)
- **Precio Objetivo 12 meses**: $X.XX (upside/downside X%)
- **Stop Loss**: $X.XX (pérdida máxima tolerable X%)
- **Catalizador clave a monitorear**: [evento específico]

**Paso 5 - Horizonte y Sizing**
- Horizonte: Corto Plazo (<3M): [Trade/No-Trade] | Largo Plazo (>12M): [Inversión/No-Inversión]
- Posición sugerida en portafolio balanceado: X% del total

**Advertencia Legal** (incluir siempre):
*Este análisis es generado por IA y no constituye asesoramiento financiero regulado. Consulte con un asesor certificado antes de tomar decisiones de inversión.*

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL. EXTENSIÓN MÁXIMA: 700 palabras.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  ETF-Specific
# ═══════════════════════════════════════════════════════════════════════════

ETF_ANALYSIS_PROMPT = """## ANÁLISIS DE ETF - {ticker}: {name}

### Datos del Fondo
| Métrica | Valor |
|---------|-------|
| Categoría | {category} |
| Activos Totales | {total_assets} |
| Expense Ratio | {expense_ratio} |
| Yield | {yield_val} |
| Retorno YTD | {ytd_return} |
| Retorno 3A | {three_year_return} |
| Retorno 5A | {five_year_return} |
| Beta (3A) | {beta_3y} |

### Top Holdings
{top_holdings}

### Pesos por Sector
{sector_weights}

---
### INSTRUCCIONES:

**Paso 1 - Eficiencia del Fondo**
- Expense Ratio: ¿Competitivo para su categoría? (benchmark: ETFs pasivos <0.10%, activos <0.50%)
- Retorno ajustado por costo: ¿El fondo entrega valor neto vs. su índice/benchmark?

**Paso 2 - Concentración y Riesgo de Diversificación**
- ¿Cuánto % representan los Top 5 holdings? ¿Hay concentración excesiva (>50%)?
- ¿Hay concentración sectorial que cree riesgo específico?

**Paso 3 - Perfil de Riesgo/Retorno**
- Evalúa el Beta {beta_3y}: ¿El ETF amplifica el mercado (>1) o lo amortigua (<1)?
- Retornos 3A vs 5A: ¿Consistencia o dependencia de un ciclo específico?

**Paso 4 - Adecuación para el Inversor**
¿Para qué tipo de inversor es apropiado?
- Perfil de riesgo: Conservador / Moderado / Agresivo
- Horizonte: Corto / Mediano / Largo plazo
- Caso de uso: Core holding / Satélite / Táctica

**Veredicto**: [COMPRAR / MANTENER / EVITAR] en 2-3 oraciones.

RESPONDE EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Financial Statements
# ═══════════════════════════════════════════════════════════════════════════

FINANCIAL_STATEMENT_PROMPT = """## ANÁLISIS DE ESTADO FINANCIERO

### Tabla de Datos
{df_input}

---
### INSTRUCCIONES:

Analiza la tabla financiera anterior siguiendo este proceso:

**Paso 1 - Tendencias Primarias**
Identifica las 3-5 líneas más importantes y su tendencia año-a-año.
Formato: "[Línea]: [X→Y→Z] = [Tendencia: Mejorando/Deteriorando/Estable]"

**Paso 2 - Alertas (Cambios >20%)**
Lista cualquier cambio interanual >20% (positivo o negativo).
Clasifica como: [POSITIVO], [NEGATIVO] o [NEUTRO EN CONTEXTO].

**Paso 3 - Indicadores de Calidad**
Para Income Statement: ¿Crecen ingresos y márgenes simultáneamente?
Para Balance Sheet: ¿Aumenta el equity? ¿Se controla la deuda?
Para Cash Flow: ¿El FCF es positivo y creciente? ¿Cuál es la calidad de los ingresos (OCF/NI)?

**Paso 4 - Resumen en 3 Líneas**
1. Fortaleza principal: [...]
2. Área de preocupación: [...]
3. Conclusión: [...]

Máximo 200 palabras. EXCLUSIVAMENTE EN ESPAÑOL.
"""
