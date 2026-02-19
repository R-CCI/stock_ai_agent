# -*- coding: utf-8 -*-
"""Institutional PDF report generation using FPDF2."""

import os
from datetime import datetime
from fpdf import FPDF
import pandas as pd
from PIL import Image

from src.config import PDF_NAVY, PDF_TEAL, PDF_SLATE, PDF_BG_GREY, PDF_TEXT_DARK


class PDFReport(FPDF):
    """Institutional investment research report PDF."""

    def __init__(self, company: str, ticker: str, **kwargs):
        super().__init__(**kwargs)
        self._company = self._sanitize(company)
        self._ticker = self._sanitize(ticker)
        self.set_auto_page_break(auto=True, margin=15)

    def _sanitize(self, text: str) -> str:
        """Replace unsupported characters with Latin-1 equivalents."""
        if not isinstance(text, str):
            return str(text)
        replacements = {
            "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
            "…": "...", "•": "-", "✔": "YES", "✘": "NO"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        if self.page_no() == 1:
            # First page header is different (Cover style)
            return

        # Professional Header for subsequent pages
        self.set_fill_color(*PDF_BG_GREY)
        self.rect(0, 0, 210, 20, "F")
        
        self.set_y(5)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*PDF_NAVY)
        self.cell(0, 10, f"{self._company} ({self._ticker})", border=0, align="L")
        
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PDF_SLATE)
        self.cell(0, 10, f"Reporte de Inversión | {datetime.now().strftime('%d %b %Y')}", border=0, align="R", ln=True)
        
        self.set_draw_color(*PDF_TEAL)
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*PDF_SLATE)
        self.cell(0, 10, f"Confidencial | Página {self.page_no()}/{{nb}}", align="C")
        self.cell(0, 10, "Stock AI Agent Institutional", align="R")

    def section_title(self, title, size=14, color=PDF_NAVY, underline=True):
        self.ln(5)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*color)
        self.cell(0, 10, self._sanitize(title), ln=True)
        if underline:
            curr_x = self.get_x()
            curr_y = self.get_y()
            self.set_draw_color(*PDF_TEAL)
            self.set_line_width(0.5)
            self.line(10, curr_y - 2, 50, curr_y - 2)
            self.set_line_width(0.2)
        self.ln(2)

    def section_body(self, text, size=10, style=""):
        self.set_font("Helvetica", style, size)
        self.set_text_color(*PDF_TEXT_DARK)
        text = str(text).replace("**", "").replace("##", "").replace("###", "")
        self.multi_cell(0, 5, self._sanitize(text))
        self.ln(2)

    def add_snapshot_box(self, metrics: dict):
        """Draw a professional 'At a Glance' metrics box."""
        self.set_fill_color(*PDF_BG_GREY)
        self.set_draw_color(*PDF_SLATE)
        x0, y0 = 10, self.get_y()
        self.rect(x0, y0, 190, 30, "FD")
        
        self.set_y(y0 + 5)
        col_w = 190 / 4
        
        # Labels
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*PDF_SLATE)
        labels = ["Precio Actual", "Cambio (%)", "Market Cap", "Sector"]
        for label in labels:
            self.cell(col_w, 5, label.upper(), align="C")
        self.ln(6)
        
        # Values
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*PDF_NAVY)
        vals = [
            f"${metrics.get('price', 'N/A')}",
            f"{round(metrics.get('change', '0'), 2)}%",
            str(metrics.get('market_cap', 'N/A')),
            str(metrics.get('sector', 'N/A'))[:15]
        ]
        for val in vals:
            self.cell(col_w, 8, val, align="C")
        self.ln(15)

    def add_dataframe(self, dataframe: pd.DataFrame):
        """Render a professional table with zebra stripes and navy header."""
        if dataframe.empty:
            return

        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*PDF_NAVY)
        self.set_text_color(255, 255, 255)
        
        cols = dataframe.columns
        # Simple width logic
        col_w = 190 / len(cols) if len(cols) > 0 else 190

        # Header
        for col in cols:
            self.cell(col_w, 8, str(col)[:15], border=1, align="C", fill=True)
        self.ln()

        # Data
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PDF_TEXT_DARK)
        fill = False
        for i, row in dataframe.iterrows():
            if fill:
                self.set_fill_color(240, 240, 240)
            else:
                self.set_fill_color(255, 255, 255)
            
            for val in row:
                # Right align if it's a number
                align = "R" if isinstance(val, (int, float)) else "L"
                self.cell(col_w, 7, str(val)[:20], border=1, align=align, fill=True)
            self.ln()
            fill = not fill
        self.ln(5)

    def add_image_safe(self, path: str, w: int = 150):
        if os.path.exists(path):
            try:
                x = (210 - w) / 2
                if self.get_y() + 80 > 270:
                    self.add_page()
                self.image(path, w=w, x=x)
                self.ln(5)
            except Exception:
                pass


def generate_report(
    ticker: str,
    overview: dict,
    results: dict,
    df_results: dict,
    conclusion: str,
    statements: dict,
    chart_paths: list | None = None,
) -> str:
    """Generate high-impact investment research report."""
    
    company = overview.get("name", ticker)
    pdf = PDFReport(company, ticker)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── COVER PAGE ──
    # Logo / Top Bar
    pdf.set_fill_color(*PDF_NAVY)
    pdf.rect(0, 0, 210, 40, "F")
    
    pdf.set_y(10)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, str(company).upper(), align="C", ln=True)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*PDF_TEAL)
    pdf.cell(0, 10, f"RESEARCH REPORT | {str(ticker).upper()}", align="C", ln=True)
    
    pdf.ln(15)
    
    # At a Glance Snapshot
    pdf.section_title("Resumen de Mercado", size=12, underline=False)
    
    price_data = results.get("current_price_data") or {}
    metrics = {
        "price": overview.get("price", 0),
        "change": price_data.get("Change", 0),
        "market_cap": overview.get("market_cap", "N/A"),
        "sector": overview.get("sector", overview.get("category", "N/A"))
    }
    pdf.add_snapshot_box(metrics)
    
    # Executive Summary / Description
    pdf.section_title("Perfil de la Empresa", size=14)
    desc = overview.get("description", "Sin descripción disponible.")
    pdf.section_body(desc[:1200] + "...")
    
    # Investment Conclusion (on front page if space)
    pdf.section_title("Tesis de Inversión (AI)", size=14)
    pdf.section_body(conclusion[:800] + " (continúa...)" if len(conclusion) > 800 else conclusion)

    # ── NEWS & MARKET ──
    if "news" in results:
        pdf.add_page()
        pdf.section_title("Análisis de Noticias y Sentimiento")
        pdf.section_body(results["news"])

    # ── VALUATION & FUNDAMENTALS ──
    if "valuation" in results or "fundamentals" in results:
        pdf.add_page()
        if "valuation" in results:
            pdf.section_title("Análisis de Valoración")
            pdf.section_body(results["valuation"])
        
        if "fundamentals" in results:
            pdf.section_title("Análisis Fundamental")
            pdf.section_body(results["fundamentals"])

    # ── DCF VALUATION ──
    if "dcf" in results:
        pdf.add_page()
        pdf.section_title("Valoración Intrínseca (DCF)")
        dcf = results["dcf"]
        g = dcf["gordon"]
        ex = dcf["exit"]
        curr_p = overview.get("price", 0)
        
        pdf.section_body("El análisis de flujos de caja descontados (DCF) proyecta los flujos libres de caja futuros y los descuenta al presente usando una tasa de retorno requerida (WACC 15%).")
        
        dcf_summary = {
            "Método de Valoración": ["Crecimiento Perpetuo (Gordon)", "Múltiplo de Salida (Exit)"],
            "Precio Intrínseco": [f"${g['implied_price']:,.2f}", f"${ex['implied_price']:,.2f}"],
            "Valor de Empresa": [f"${g['enterprise_value']:,.2f}", f"${ex['enterprise_value']:,.2f}"],
            "% del Valor en Terminal": [f"{g['pct_from_tv']}%", f"{ex['pct_from_tv']}%"],
            "Margen de Seguridad": [
                f"{max(0, (1 - curr_p / g['implied_price'])*100):.1f}%" if g['implied_price'] > 0 else "0%",
                f"{max(0, (1 - curr_p / ex['implied_price'])*100):.1f}%" if ex['implied_price'] > 0 else "0%"
            ]
        }
        pdf.add_dataframe(pd.DataFrame(dcf_summary))
        
        pdf.section_body(f"Nota: Se asume un crecimiento perpetuo del 2% y un múltiplo EBITDA de salida de 12x para el escenario base. El precio actual de mercado es ${curr_p:,.2f}.")

    # ── FINANCIALS ──
    if statements and overview.get("instrument_type") != "ETF":
        pdf.add_page()
        pdf.section_title("Estados Financieros")
        for label, key in [("Estado de Resultados", "income"), ("Balance General", "balance"), ("Flujo de Caja", "cashflow")]:
            if key in statements and not statements[key].empty:
                pdf.section_title(label, size=11, color=PDF_SLATE, underline=False)
                # Take top 5 rows for the report to keep it clean
                pdf.add_dataframe(statements[key].head(10))
                if key in df_results:
                     pdf.section_body(df_results[key], size=9)

    # ── RISK & MONTE CARLO ──
    if "risk" in results or "montecarlo" in results:
        pdf.add_page()
        if "risk" in results:
            pdf.section_title("Análisis de Riesgo")
            pdf.section_body(results["risk"])
        
        if "montecarlo" in results:
            pdf.section_title("Simulación Monte Carlo")
            pdf.section_body(results["montecarlo"])

    # ── OPTIONS ──
    if "options" in results:
        pdf.add_page()
        pdf.section_title("Análisis de Opciones")
        pdf.section_body(results["options"])

    # ── CHARTS ──
    if chart_paths:
        pdf.add_page()
        pdf.section_title("Análisis Técnico y Gráficos")
        for path in chart_paths:
            pdf.add_image_safe(path)
        
        if "technicals" in results:
            pdf.section_body(results["technicals"])

    # ── FINAL CONCLUSION & DISCLAIMER ──
    pdf.add_page()
    pdf.section_title("Conclusión Detallada")
    pdf.section_body(conclusion)
    
    pdf.ln(20)
    pdf.set_draw_color(*PDF_SLATE)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*PDF_SLATE)
    disclaimer = ("AVISO LEGAL: Este informe es generado utilizando AI, y no constituye asesoramiento financiero ni una recomendación de compra o venta. Invertir conlleva riesgos significativos.")
    pdf.multi_cell(0, 4, disclaimer)

    # Save
    filename = f"{ticker}_Equity_Research_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(filename)
    return filename

