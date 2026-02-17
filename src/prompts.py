# -*- coding: utf-8 -*-
"""Professional LLM prompts for all analysis modules (SPANISH VERSION)."""


# ═══════════════════════════════════════════════════════════════════════════
#  System Prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Eres un Analista de Inversiones Senior (CFA/CMT) en un fondo de cobertura de primer nivel.
Tu objetivo es proporcionar análisis de inversión de grado institucional basados en datos.
INSTRUCCIÓN CRÍTICA: Todo tu análisis, razonamiento y salida DEBE ser en ESPAÑOL.

## Principios Fundamentales
1. **Basado en Datos**: Nunca afirmes nada sin citar los datos proporcionados.
2. **Visión Equilibrada**: Analiza siempre los casos Alcista (Bull) y Bajista (Bear).
3. **Tono Profesional**: Usa terminología financiera precisa en español. Evita jerga de retail (ej. "to the moon").
4. **Conciencia del Riesgo**: Enfatiza los riesgos, drawdowns y volatilidad junto con los retornos potenciales.
5. **Estructura**: Usa encabezados claros, viñetas y párrafos cortos.

Tu audiencia confía en este análisis para decisiones de asignación de capital multimillonarias.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Valuation
# ═══════════════════════════════════════════════════════════════════════════

VALUATION_PROMPT = """ANÁLISIS DE VALORACIÓN: {ticker} ({instrument_type})

DATOS PROPORCIONADOS:
Precio Actual: {price}
Precio Objetivo: {target_price}
Industria: {industry}

Ratios de Valoración vs. Promedios de la Industria:
- P/E: {pe_ratio} (Ind: {industry_pe_avg})
- Forward P/E: {forward_pe_ratio}
- PEG: {peg_ratio} (Ind: {industry_peg_avg})
- P/S: {ps_ratio} (Ind: {industry_ps_avg})
- P/B: {pb_ratio} (Ind: {industry_pb_avg})
- P/C: {pc_ratio} (Ind: {industry_pc_avg})
- P/FCF: {pfcf_ratio} (Ind: {industry_pfcf_avg})

Crecimiento (EPS/Ventas):
- EPS este año: {eps_this_y}
- EPS próximo año: {eps_next_y}
- EPS pasados 5 años: {eps_past_5y}
- EPS próximos 5 años: {eps_next_5y}
- Ventas pasados 5 años: {sales_past_5y}

INSTRUCCIONES:
1. Evalúa si la acción está infravalorada, sobrevalorada o justamente valorada basándote en los múltiplos (P/E, PEG, P/B) comparados con la industria.
2. Analiza las perspectivas de crecimiento (EPS y Ventas). ¿Justifican la valoración actual?
3. Para REITs/ETFs, enfócate en P/FFO o el NAV si aplica (o usa los métricos disponibles como proxies).
4. Concluye con un Veredicto de Valoración (ej. "Atractiva", "Neutral", "Costosa").

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Fundamentals
# ═══════════════════════════════════════════════════════════════════════════

FUNDAMENTALS_PROMPT = """ANÁLISIS FUNDAMENTAL: {ticker}

Propiedad e Interés en Corto:
- Insider Own: {insider_ownership}
- Inst Own: {institutions_own}
- Inst Trans: {institutions_trans}
- Short Float: {short_float}
- Short Ratio: {short_ratio}

Salud Financiera & Márgenes:
- Market Cap: {market_cap}
- Margen Bruto: {gross_margin} | Margen Operativo: {oper_margin} | Margen Neto: {profit_margin}
- ROA: {roa} | ROE: {roe} | ROI: {roic}
- Quick Ratio: {quick_ratio} | Current Ratio: {current_ratio}
- Deuda/Eq: {debt_eq} | LT Debt/Eq: {lt_debt_eq}

INSTRUCCIONES:
1. Analiza la calidad del negocio a través de sus márgenes (¿son saludables/líderes?).
2. Evalúa la eficiencia del capital (ROE/ROIC).
3. Evalúa la salud del balance (Niveles de deuda y liquidez).
4. Comenta sobre la "Smart Money" (Institucionales) y el sentimiento bajista (Short Float).

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Technical Analysis
# ═══════════════════════════════════════════════════════════════════════════

TECHNICAL_PROMPT = """ANÁLISIS TÉCNICO: {ticker}

DATOS:
Precios:
- SMA20: {sma_20} | SMA50: {sma_50} | SMA200: {sma_200}
- 52W Low: {low} | 52W High: {high}

Indicadores de Momento & Volatilidad:
- RSI (14): {rsi_14}
- ATR: {atr_14}
- Volatilidad (W/M): {volatility_w} / {volatility_m}

Patrones Detectados:
{detected_patterns}

Resumen de Indicadores:
{ta_summary}

INSTRUCCIONES:
1. Determina la tendencia principal (Alcista, Bajista, Lateral) basándote en las SMAs.
2. Analiza el momento (RSI) y la volatilidad (ATR/Bandas).
3. Evalúa la importancia de los patrones detectados.
4. Identifica niveles clave de soporte y resistencia implícitos.
5. Conclusión Técnica: ¿Es un buen punto de entrada/salida?

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Risk Metrics
# ═══════════════════════════════════════════════════════════════════════════

RISK_METRICS_PROMPT = """ANÁLISIS DE RIESGO: {ticker} vs Benchmark ({benchmark})

Métricas Calculadas:
- Retorno Anualizado: {annualized_return}
- Volatilidad (Desv. Est. Anual): {vol}
- Volatilidad a la Baja: {downside_vol}
- Max Drawdown: {max_dd}
- Beta: {beta}
- Sharpe Ratio: {sharpe_ratio}
- Sortino Ratio: {sortino_ratio}
- Treynor Ratio: {treynor_ratio}
- Calmar Ratio: {calmar_ratio}

Tasa Libre de Riesgo usada: {rfr}%

INSTRUCCIONES:
1. Interpreta el Sharpe y Sortino. ¿Compensa el retorno por el riesgo asumido?
2. Analiza el Beta. ¿Qué tan sensible es al mercado?
3. Evalúa el riesgo de cola (Max Drawdown). ¿Es tolerable para un inversor conservador?
4. Proporciona una Evaluación de Riesgo (Bajo, Moderado, Alto, Especulativo).

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════

MONTECARLO_PROMPT = """SIMULACIÓN MONTE CARLO (GMM): {ticker}

Configuración: {n_simulations} simulaciones a {days} días.
Precio Actual: {last_price}

Resultados (Proyección de Precios al Final del Periodo):
{percentile_summary}

INSTRUCCIONES:
1. Interpreta el rango probable de precios (Percentiles 25-75).
2. Analiza los escenarios extremos (P5 y P95). ¿Cuál es el riesgo a la baja vs. potencial al alza?
3. Define la asimetría (skewness) de los resultados. ¿Hay más probabilidad de alza o baja?
4. Conclusión basada en probabilidades.

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Options
# ═══════════════════════════════════════════════════════════════════════════

OPTIONS_PROMPT = """ANÁLISIS DE OPCIONES: {ticker}

Datos del Mercado de Opciones:
- Precio Spot: {spot_price}
- Expiración Analizada: {expiration} (Días: {days_to_expiry})
- IV Aprox (ATM): {iv_format}
- Movimiento Esperado (Implied Move): ±{expected_move_return}
- Rango Implícito: {range_lower_bound} - {range_upper_bound}
- Max Pain: {max_pain_strike} (Distancia: {max_pain_distance})

Estructura de Mercado (Sentiment):
- Put/Call OI Ratio: {put_call_oi_ratio}
- Put/Call Vol Ratio: {put_call_volume_ratio}
- Total Call OI: {total_call_oi}
- Total Put OI: {total_put_oi}

Cadenas (Calls/Puts Principales):
{options_data}

INSTRUCCIONES:
1. Analiza el sentimiento del mercado de opciones (Put/Call ratios, posicionamiento).
2. Interpreta el 'Max Pain' y su posible efecto magnético en el precio.
3. Evalúa la Volatilidad Implícita (IV). ¿Está barata o cara? (Si es relevante).
4. Comenta sobre los muros de Call/Put OI si son visibles.

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Macro Overview
# ═══════════════════════════════════════════════════════════════════════════

MACRO_PROMPT = """INFORME MACROECONÓMICO

Resumen de Artículos/Blogs Recientes:
{macro_articles}

INSTRUCCIONES:
1. Sintetiza los temas clave que mueven el mercado (Inflación, Fed/Tasas, Crecimiento, Geopolítica).
2. Identifica vientos de cola (positivos) y vientos en contra (negativos) para el mercado de renta variable en general.
3. Proporciona una perspectiva de corto plazo para el entorno de inversión ("Risk-On" o "Risk-Off").

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Market News
# ═══════════════════════════════════════════════════════════════════════════

MARKET_NEWS_PROMPT = """ANÁLISIS DE SENTIMIENTO DE MERCADO

Índice de Miedo y Codicia (Fear & Greed): {fear_greed}
Rendimiento del SPY (Benchmark):
- Día: {perf_day}% | Semana: {perf_week}% | Mes: {perf_month}%
- YTD: {perf_ytd}%

Volatilidad del SPY:
- Semana: {vol_week}% | Mes: {vol_month}%

Titulares de Noticias de Mercado:
{news}

INSTRUCCIONES:
1. Determina el "Humor del Mercado" actual (e.g., Miedo Extremo, Cautela, Euforia).
2. Analiza si las noticias apoyan la tendencia actual de precios.
3. Identifica riesgos sistémicos mencionados en los titulares.

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Stock-Specific News
# ═══════════════════════════════════════════════════════════════════════════

STOCK_NEWS_PROMPT = """ANÁLISIS DE NOTICIAS CORPORATIVAS: {ticker}

Titulares Recientes:
{news}

INSTRUCCIONES:
1. Identifica los catalizadores más importantes (Ganancias, Productos, Fusiones, Regulaciones).
2. Evalúa el sentimiento agregado de las noticias (Positivo, Negativo, Mixto).
3. Destaca cualquier "Bandera Roja" o riesgo inminente mencionado.

Responde EXCLUSIVAMENTE en Español.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Conclusion / Final Recommendation
# ═══════════════════════════════════════════════════════════════════════════

CONCLUSION_PROMPT = """TESIS DE INVERSIÓN FINAL: {company} ({ticker})

SÍNTESIS DE ANÁLISIS PREVIOS:
1. Valoración: {valuation}
2. Fundamentos: {fundamentals}
3. Técnico: {technicals}
4. Riesgo: {risk_metrics}
5. Monte Carlo: {gmm_montecarlo}
6. Sentimiento/Noticias: {news}
7. Opciones (Si aplica): {options_short_term_analysis}

INSTRUCCIONES:
Actúa como un Portfolio Manager Senior y redacta la Tesis de Inversión Final.
1. **Veredicto Ejecutivo**: (Compra Fuerte / Compra / Mantener / Venta / Venta Fuerte).
2. **Razonamiento (El "Por qué")**: Sintetiza los puntos más fuertes a favor y en contra. Conecta los fundamentales con los técnicos.
3. **Plan de Acción**:
   - Rango de entrada sugerido (basado en técnicos/valoración).
   - Precio Objetivo (Target) a 12 meses.
   - Nivel de Stop Loss sugerido (gestión de riesgo).
4. **Horizonte Temporal**: Corto Plazo (Trade) vs Largo Plazo (Inversión).

Escribe de forma persuasiva, profesional y directa. EXCLUSIVAMENTE EN ESPAÑOL.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  ETF-Specific
# ═══════════════════════════════════════════════════════════════════════════

ETF_ANALYSIS_PROMPT = """ANÁLISIS DE ETF: {ticker} - {name}

Categoría: {category}
Activos Totales: {total_assets}
Expense Ratio: {expense_ratio}
Yield: {yield_val}

Rendimiento:
- YTD: {ytd_return}
- 3 Años: {three_year_return}
- 5 Años: {five_year_return}
- Beta (3Y): {beta_3y}

Top Holdings:
{top_holdings}

Pesos por Sector:
{sector_weights}

INSTRUCCIONES:
1. Analiza la estructura del fondo (costos vs. rendimiento).
2. Evalúa la diversificación y concentración (Top Holdings y Sectores).
3. Identifica la exposición al riesgo (Beta y tipos de activos).
4. Veredicto: ¿Es un buen vehículo para exposición a este sector/estrategia?

Responde EXCLUSIVAMENTE en Español.
"""

FINANCIAL_STATEMENT_PROMPT = """
Analiza la siguiente tabla financiera (Income/Balance/CashFlow) para identificar tendencias clave, fortalezas y debilidades.
Enfócate en cambios porcentuales grandes, márgenes y salud financiera. Sé conciso.
Responde EXCLUSIVAMENTE en Español.

Datos:
{df_input}
"""
