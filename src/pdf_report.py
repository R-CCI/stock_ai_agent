# -*- coding: utf-8 -*-
"""
Professional Equity Research PDF Report — Institutional Grade Design.

Layout inspired by Goldman Sachs / Morgan Stanley equity research format.
Uses FPDF2 with multi-column layouts, color-coded sections, and data tables.
"""

import os
from datetime import datetime
from fpdf import FPDF
import pandas as pd

from src.config import PDF_NAVY, PDF_TEAL, PDF_SLATE, PDF_BG_GREY, PDF_TEXT_DARK

CCI_CHARCOAL = (68, 66, 69)
CCI_GREEN = (163, 198, 48)
CCI_GOLD = (245, 212, 92)
CCI_SOFT_GREY = (239, 239, 239)
CCI_TEXT = (52, 52, 56)

# ── Color palette ──────────────────────────────────────────────────────────
PDF_NAVY = CCI_CHARCOAL
PDF_TEAL = CCI_GREEN
PDF_SLATE = (112, 112, 118)
PDF_BG_GREY = CCI_SOFT_GREY
PDF_TEXT_DARK = CCI_TEXT

_GREEN = CCI_GREEN
_RED = (190, 58, 53)
_AMBER = CCI_GOLD
_BLUE = (207, 215, 104)
_WHITE = (255, 255, 255)
_LIGHT = (250, 250, 247)
_LIGHT_GREEN = (245, 249, 232)
_LIGHT_GOLD = (252, 248, 228)
_BORDER = (210, 210, 205)


# ═══════════════════════════════════════════════════════════════════════════
#  Core PDF Class
# ═══════════════════════════════════════════════════════════════════════════

class PDFReport(FPDF):
    """Institutional equity research PDF with professional design."""

    def __init__(self, company: str, ticker: str, **kwargs):
        super().__init__(**kwargs)
        self._company = self._sanitize(company)
        self._ticker  = self._sanitize(ticker.upper())
        self._date    = datetime.now().strftime("%d %B %Y")
        self.set_auto_page_break(auto=True, margin=20)

    # ── Utilities ─────────────────────────────────────────────────────────

    def _sanitize(self, text: str) -> str:
        """Normalize text to Latin-1 safe representation — strips all non-printable and non-Latin-1 chars."""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        replacements = {
            # Smart quotes
            "\u2018": "'", "\u2019": "'", "\u201a": "'",
            "\u201c": '"', "\u201d": '"', "\u201e": '"',
            # Dashes — CRITICAL for PDF
            "\u2013": "-", "\u2014": "-", "\u2015": "-",
            "\u2212": "-",
            # Ellipsis
            "\u2026": "...",
            # Bullets / special chars
            "\u2022": "-", "\u2023": "-", "\u25e6": "-",
            "\u2713": "OK", "\u2717": "X", "\u2714": "OK",
            # Common symbols
            "\u00b0": " deg", "\u00b1": "+/-", "\u00d7": "x", "\u00f7": "/",
            "\u2192": "->", "\u2190": "<-", "\u2191": "^", "\u2193": "v",
            "\u00a9": "(c)", "\u00ae": "(R)", "\u2122": "(TM)",
            # Emoji / pictographs — replace with empty or short tag
            "\u2705": "[OK]", "\u274c": "[X]", "\u2b50": "*",
            "\u2764": "<3", "\u25cf": "-", "\u2b24": "-",
            "\u26a0": "[!]",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        # Strip any remaining non-Latin-1 characters (emoji, CJK, etc.)
        # encode with 'replace' turns them into '?' — then decode back
        text = text.encode("latin-1", "replace").decode("latin-1")

        # Remove lone '?' artifacts from emoji sequences (optional cleanup)
        import re
        text = re.sub(r'\?{3,}', '...', text)
        return text

    def _set_color(self, rgb: tuple, fill=False, draw=False, text=False):
        if fill:  self.set_fill_color(*rgb)
        if draw:  self.set_draw_color(*rgb)
        if text:  self.set_text_color(*rgb)

    def _rule(self, color=PDF_TEAL, thickness=0.4):
        self._set_color(color, draw=True)
        self.set_line_width(thickness)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_line_width(0.2)
        self._set_color(_BORDER, draw=True)

    def _accent_bar(self, y: float, color: tuple, h: float = 2.0):
        self._set_color(color, fill=True)
        self.rect(10, y, 190, h, "F")

    # ── Page Decorators ──────────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return

        # Top bar
        self._set_color(PDF_NAVY, fill=True)
        self.rect(0, 0, 210, 16, "F")
        self._set_color(CCI_GOLD, fill=True)
        self.rect(0, 0, 210, 1.6, "F")

        self.set_y(4)
        self.set_font("Helvetica", "B", 8.5)
        self._set_color(_WHITE, text=True)
        self.cell(105, 8, f"  {self._company} ({self._ticker})", align="L")
        self.set_font("Helvetica", "", 7.5)
        self.cell(0, 8, f"CCI Puesto de Bolsa  |  {self._date}  ", align="R", ln=True)

        self._set_color(PDF_TEAL, draw=True)
        self.set_line_width(1.1)
        self.line(0, 16, 210, 16)
        self.set_line_width(0.2)
        self.set_y(20)

    def footer(self):
        self.set_y(-12)
        self._set_color(_LIGHT, fill=True)
        self.rect(0, self.get_y(), 210, 12, "F")
        self._set_color(CCI_GOLD, draw=True)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_line_width(0.2)
        self.set_font("Helvetica", "I", 7)
        self._set_color(PDF_SLATE, text=True)
        self.cell(
            0, 8,
            f"CONFIDENCIAL - Uso exclusivo de CCI Puesto de Bolsa  |  Pag. {self.page_no()}/{{nb}}  |"
            f"  Este documento no constituye asesoramiento financiero regulado.",
            align="C",
        )

    # ── Layout Primitives ────────────────────────────────────────────────

    def section_title(self, title: str, size: int = 13, color=PDF_NAVY):
        self.ln(4)
        y = self.get_y()
        self._set_color(_LIGHT, fill=True)
        self._set_color(_BORDER, draw=True)
        self.rect(10, y, 190, 10, "FD")
        self._set_color(CCI_GREEN, fill=True)
        self.rect(10, y, 6, 10, "F")
        self.set_xy(19, y + 1)
        self.set_font("Helvetica", "B", size)
        self._set_color(color, text=True)
        self.cell(0, 8, self._sanitize(title), ln=True)
        self._accent_bar(y + 10.8, CCI_GOLD, 1.3)
        self.ln(5)

    def subsection(self, title: str, size: int = 10):
        self.ln(2)
        self.set_font("Helvetica", "B", size)
        self._set_color(PDF_NAVY, text=True)
        self.cell(0, 6, self._sanitize(title), ln=True)
        self._accent_bar(self.get_y(), CCI_GREEN, 1.1)
        self.ln(1)

    def body_text(self, text: str, size: int = 9, indent: int = 0):
        """Render body text, stripping markdown formatting."""
        text = str(text or "")
        # Strip markdown
        for ch in ["**", "*", "##", "###", "# "]:
            text = text.replace(ch, "")
        text = text.strip()
        self.set_font("Helvetica", "", size)
        self._set_color(PDF_TEXT_DARK, text=True)
        if indent:
            self.set_x(10 + indent)
        self.multi_cell(0, 5, self._sanitize(text))
        self.ln(2)

    def colored_cell(self, label: str, value: str, label_color=PDF_NAVY, val_color=PDF_TEXT_DARK, w=45):
        """Mini key-value cell pair."""
        self.set_font("Helvetica", "B", 7.5)
        self._set_color(label_color, text=True)
        self.cell(w, 5, self._sanitize(label.upper()), align="L")
        self.set_font("Helvetica", "", 8.5)
        self._set_color(val_color, text=True)
        self.cell(w, 5, self._sanitize(str(value)), align="L")

    # ── Complex Components ───────────────────────────────────────────────

    def rating_badge(self, rating: str, x: float = 150, y: float = None):
        """Draw a color-coded rating badge."""
        y = y or self.get_y()
        rating_upper = rating.upper()
        color_map = {
            "COMPRA FUERTE": _GREEN,
            "COMPRA":        _GREEN,
            "MANTENER":      _AMBER,
            "VENTA":         _RED,
            "VENTA FUERTE":  _RED,
        }
        bg = color_map.get(rating_upper, PDF_TEAL)
        self.set_xy(x, y)
        self._set_color(bg, fill=True)
        self._set_color(bg, draw=True)
        self.rect(x, y, 50, 10, "F")
        self.set_xy(x, y + 1)
        self.set_font("Helvetica", "B", 9)
        self._set_color(_WHITE, text=True)
        self.cell(50, 8, rating_upper, align="C")
        self.set_xy(10, y + 14)

    def kpi_row(self, items: list, col_width: float = 38):
        """
        Render a horizontal KPI strip.
        items: list of (label, value, delta, delta_positive)
        """
        y_start = self.get_y()
        self._set_color(_LIGHT, fill=True)
        self._set_color(_BORDER, draw=True)
        n = len(items)
        total_w = col_width * n
        x_start = (210 - total_w) / 2

        for i, (label, value, delta, pos) in enumerate(items):
            x = x_start + i * col_width
            self.rect(x, y_start, col_width - 2, 20, "FD")
            accent = CCI_GREEN if i % 2 == 0 else CCI_GOLD
            self._set_color(accent, fill=True)
            self.rect(x, y_start, col_width - 2, 2, "F")
            # Label
            self.set_xy(x + 1, y_start + 4)
            self.set_font("Helvetica", "B", 6.5)
            self._set_color(PDF_SLATE, text=True)
            self.cell(col_width - 4, 4, self._sanitize(label.upper()), align="C")
            # Value
            self.set_xy(x + 1, y_start + 9)
            self.set_font("Helvetica", "B", 10)
            self._set_color(PDF_NAVY, text=True)
            self.cell(col_width - 4, 5, self._sanitize(str(value)), align="C")
            # Delta
            if delta:
                self.set_xy(x + 1, y_start + 15)
                self.set_font("Helvetica", "", 7)
                d_color = _GREEN if pos else _RED
                self._set_color(d_color, text=True)
                self.cell(col_width - 4, 4, self._sanitize(str(delta)), align="C")

        self.set_y(y_start + 23)

    def data_table(
        self,
        df: pd.DataFrame,
        max_rows: int = 12,
        col_widths: list | None = None,
        font_size: int = 7,
    ):
        """Render a professionally styled data table with zebra stripes."""
        if df is None or df.empty:
            return

        df = df.head(max_rows)
        cols = list(df.columns)
        n_cols = len(cols)
        if n_cols == 0:
            return

        page_w = 190
        if col_widths is None:
            # First column wider for labels
            first_w = min(60, page_w // 3)
            rest_w  = (page_w - first_w) / max(n_cols - 1, 1)
            col_widths = [first_w] + [rest_w] * (n_cols - 1)

        # Header
        self._set_color(PDF_NAVY, fill=True)
        self._set_color(_WHITE, text=True)
        self.set_font("Helvetica", "B", font_size)
        for j, col in enumerate(cols):
            self.cell(col_widths[j], 7, self._sanitize(str(col))[:22], border=1, align="C", fill=True)
        self.ln()

        # Rows
        self._set_color(PDF_TEXT_DARK, text=True)
        for i, (_, row) in enumerate(df.iterrows()):
            fill_color = _LIGHT_GREEN if i % 2 == 0 else _WHITE
            self._set_color(fill_color, fill=True)
            self.set_fill_color(*fill_color)
            self.set_font("Helvetica", "", font_size)
            for j, val in enumerate(row):
                align = "R" if isinstance(val, (int, float)) else "L"
                self.cell(col_widths[j], 6, self._sanitize(str(val))[:28], border=1, align=align, fill=True)
            self.ln()

        self.ln(4)

    def two_col_text(self, left: str, right: str, title_l: str = "", title_r: str = ""):
        """Render two text blocks side by side."""
        col_w = 90
        y0 = self.get_y()

        # Left
        self.set_xy(10, y0)
        if title_l:
            self.set_font("Helvetica", "B", 9)
            self._set_color(PDF_NAVY, text=True)
            self.cell(col_w, 6, self._sanitize(title_l), ln=False)
            self.set_xy(10, self.get_y() + 6)
        self.set_font("Helvetica", "", 8)
        self._set_color(PDF_TEXT_DARK, text=True)
        self.multi_cell(col_w, 4.5, self._sanitize(str(left)[:600]))
        y_after_left = self.get_y()

        # Right
        self.set_xy(110, y0)
        if title_r:
            self.set_font("Helvetica", "B", 9)
            self._set_color(PDF_NAVY, text=True)
            self.cell(col_w, 6, self._sanitize(title_r), ln=False)
            self.set_xy(110, y0 + 6)
        self.set_font("Helvetica", "", 8)
        self._set_color(PDF_TEXT_DARK, text=True)
        # FPDF multi_cell starts at current x, so we need to set x before each line
        lines = str(right)[:600].split("\n")
        for line in lines:
            self.set_x(110)
            self.cell(col_w, 4.5, self._sanitize(line), ln=True)

        self.set_y(max(y_after_left, self.get_y()) + 5)

    def scenario_table(self, scenarios: dict, current_price: float):
        """Render bull/base/bear scenario comparison table."""
        self.subsection("Análisis de Escenarios DCF")

        headers = ["Escenario", "Crec. S1", "Crec. S2", "WACC", "Precio Gordon", "Precio Exit", "Upside/Down"]
        widths  = [32, 22, 22, 22, 28, 28, 30]

        # Header row
        self._set_color(PDF_NAVY, fill=True)
        self._set_color(_WHITE, text=True)
        self.set_font("Helvetica", "B", 7)
        for h, w in zip(headers, widths):
            self.cell(w, 7, h, border=1, align="C", fill=True)
        self.ln()

        color_map = {"bull": _GREEN, "base": _AMBER, "bear": _RED}
        self.set_font("Helvetica", "", 7.5)
        for key, s in scenarios.items():
            bg = color_map.get(key, _LIGHT)
            self.set_fill_color(*bg)
            self._set_color(PDF_NAVY if key == "base" else _WHITE, text=True)

            g_price = s.get("gordon_price", 0)
            e_price = s.get("exit_price", 0)
            upside  = f"{(g_price/current_price-1)*100:+.1f}%" if current_price > 0 else "N/D"

            vals = [
                s.get("label", key),
                f"{s.get('stage1_growth',0)*100:.0f}%",
                f"{s.get('stage2_growth',0)*100:.0f}%",
                f"{s.get('wacc',0)*100:.1f}%",
                f"${g_price:,.2f}",
                f"${e_price:,.2f}",
                upside,
            ]
            for v, w in zip(vals, widths):
                self.cell(w, 6.5, self._sanitize(v), border=1, align="C", fill=True)
            self.ln()

        self.ln(5)

    def add_image_safe(self, path: str, w: int = 170, caption: str = ""):
        """Insert a chart image with optional caption."""
        if not path or not os.path.exists(path):
            return
        try:
            x = (210 - w) / 2
            if self.get_y() + 85 > self.h - 20:
                self.add_page()
            self.image(path, x=x, w=w)
            if caption:
                self.set_font("Helvetica", "I", 7)
                self._set_color(PDF_SLATE, text=True)
                self.cell(0, 5, self._sanitize(caption), align="C", ln=True)
            self.ln(4)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Report Assembly
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    ticker: str,
    overview: dict,
    results: dict,
    df_results: dict,
    conclusion: str,
    statements: dict,
    chart_paths: list | None = None,
    dcf_scenarios: dict | None = None,
) -> str:
    """
    Assemble a professional institutional equity research PDF.

    Sections:
      1. Cover Page
      2. Executive Summary & Investment Rating
      3. Company Overview
      4. Market Sentiment & News
      5. Valuation Analysis
      6. Fundamental Analysis
      7. DCF Valuation (with Scenarios)
      8. Financial Statements
      9. Technical Analysis + Charts
      10. Risk Profile
      11. Monte Carlo Simulation
      12. Options Analysis (if applicable)
      13. Investment Thesis & Rating
      14. Legal Disclaimer
    """
    company = overview.get("name", ticker) or ticker
    pdf = PDFReport(company, ticker)
    pdf.alias_nb_pages()

    price_data    = results.get("current_price_data") or {}
    current_price = float(overview.get("price") or price_data.get("Price") or 0)
    change_pct    = float(price_data.get("Change") or 0)
    market_cap    = str(overview.get("market_cap") or "N/D")
    sector        = str(overview.get("sector") or overview.get("category") or "N/D")
    instrument    = str(overview.get("instrument_type") or "Stock")

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 1 — COVER
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()

    pdf.set_fill_color(*_LIGHT)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_fill_color(*PDF_NAVY)
    pdf.rect(0, 0, 210, 66, "F")
    pdf.set_fill_color(*CCI_GOLD)
    pdf.rect(0, 66, 210, 3, "F")
    pdf.set_fill_color(*PDF_TEAL)
    pdf.rect(0, 69, 210, 2, "F")

    pdf.set_xy(14, 10)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_WHITE)
    pdf.cell(0, 5, "CCI PUESTO DE BOLSA", ln=True)

    pdf.set_xy(14, 19)
    pdf.set_font("Helvetica", "B", 34)
    pdf.set_text_color(*_WHITE)
    pdf.cell(12, 10, "C", ln=False)
    pdf.set_text_color(*PDF_TEAL)
    pdf.cell(12, 10, "C", ln=False)
    pdf.set_text_color(*CCI_GOLD)
    pdf.cell(12, 10, "I", ln=False)

    pdf.set_xy(58, 22)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*_WHITE)
    pdf.cell(0, 8, "PUESTO DE BOLSA", ln=True)
    pdf.set_x(58)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*PDF_TEAL)
    pdf.cell(0, 6, "MIEMBRO DE LA BVRD", ln=True)

    pdf.set_xy(14, 86)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*PDF_NAVY)
    pdf.cell(182, 12, company.upper(), align="C", ln=True)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*PDF_TEAL)
    pdf.cell(182, 8, f"{ticker.upper()}  |  EQUITY RESEARCH REPORT", align="C", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*PDF_SLATE)
    pdf.cell(182, 6, f"Generado: {datetime.now().strftime('%d %B %Y  %H:%M')}  |  "
             f"Instrumento: {instrument}  |  Moneda: {overview.get('currency', 'USD')}",
             align="C", ln=True)

    # ── KPI Strip ──
    pdf.set_y(110)
    change_str = f"{change_pct:+.2f}%" if change_pct else "N/D"
    kpis = [
        ("Precio Actual",  f"${current_price:,.2f}" if current_price else "N/D", change_str, change_pct >= 0),
        ("Market Cap",     market_cap,     "",    True),
        ("Sector",         sector[:14],    "",    True),
        ("Exchange",       str(overview.get("exchange") or "N/D"), "", True),
        ("Empleados",      str(overview.get("employees") or "N/D"), "", True),
    ]
    pdf.kpi_row(kpis, col_width=38)

    # ── Investment Rating (extracted from conclusion) ──
    rating = _extract_rating(conclusion)
    if rating:
        pdf.set_y(pdf.get_y() + 4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*PDF_SLATE)
        pdf.cell(0, 6, "RATING DE INVERSIÓN:", align="C", ln=True)
        pdf.rating_badge(rating, x=80, y=pdf.get_y())
        pdf.ln(14)

    # ── Executive Summary Box ──
    pdf.ln(6)
    pdf.set_fill_color(*_LIGHT_GOLD)
    pdf.set_draw_color(*_BORDER)
    pdf.set_line_width(0.3)
    y_box = pdf.get_y()
    pdf.rect(10, y_box, 190, 45, "FD")
    pdf.set_fill_color(*PDF_TEAL)
    pdf.rect(10, y_box, 190, 5, "F")
    pdf.set_xy(14, y_box + 4)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_WHITE)
    pdf.cell(0, 5, "RESUMEN EJECUTIVO", ln=True)
    pdf.set_xy(14, y_box + 11)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*PDF_TEXT_DARK)
    summary = _extract_thesis_summary(conclusion)
    pdf.multi_cell(182, 4.5, pdf._sanitize(summary[:450]))

    # ── Table of Contents (cosmetic) ──
    pdf.set_y(y_box + 54)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*PDF_NAVY)
    pdf.cell(0, 5, "CONTENIDO DEL REPORTE:", ln=True)
    toc_items = [
        "1. Perfil de la Empresa",
        "2. Análisis de Mercado y Noticias",
        "3. Valoración (Múltiplos)",
        "4. Análisis Fundamental",
        "5. Valoración DCF / Intrínseca",
        "6. Estados Financieros",
        "7. Análisis Técnico",
        "8. Perfil de Riesgo",
        "9. Simulación Monte Carlo",
        "10. Análisis de Opciones",
        "11. Tesis de Inversión Final",
    ]
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*PDF_TEXT_DARK)
    cols = [toc_items[:6], toc_items[6:]]
    for col_idx, col in enumerate(cols):
        x = 14 + col_idx * 96
        for i, item in enumerate(col):
            pdf.set_xy(x, pdf.get_y() if col_idx == 0 else y_box + 58 + i * 5)
            if col_idx == 0:
                pdf.cell(0, 5, f"   {item}", ln=True)

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 2 — COMPANY OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("1. Perfil de la Empresa")

    desc = str(overview.get("description") or "Descripción no disponible.")
    pdf.body_text(desc[:1800])

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 3 — MARKET SENTIMENT & NEWS
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("2. Análisis de Mercado y Sentimiento")

    if results.get("news"):
        pdf.subsection(f"Noticias Corporativas — {ticker}")
        pdf.body_text(results["news"])

    if results.get("market") or results.get("macro"):
        pdf.subsection("Contexto de Mercado y Macro")
        _add_two_col(pdf, results.get("market", ""), results.get("macro", ""),
                     "Sentimiento de Mercado", "Entorno Macroeconómico")

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 4 — VALUATION & FUNDAMENTALS
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("3. Análisis de Valoración")

    if results.get("valuation"):
        pdf.body_text(results["valuation"])

    if results.get("fundamentals"):
        pdf.section_title("4. Análisis Fundamental")
        pdf.body_text(results["fundamentals"])

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 5 — DCF VALUATION
    # ══════════════════════════════════════════════════════════════════════
    if "dcf" in results:
        pdf.add_page()
        pdf.section_title("5. Valoración por Flujos Descontados (DCF)")

        pdf.body_text(
            "El modelo DCF descuenta los flujos libres de caja proyectados al presente usando "
            "una tasa de retorno requerida (WACC). Se presentan dos métodos de valor terminal: "
            "Crecimiento Perpetuo (Gordon Growth) y Múltiplo de Salida (EV/EBITDA)."
        )

        dcf = results["dcf"]
        g, ex = dcf.get("gordon", {}), dcf.get("exit", {})

        # DCF Summary Table
        dcf_df = pd.DataFrame({
            "Método": ["Gordon Growth (Perpetuo)", "Exit Multiple (EV/EBITDA)"],
            "Precio Intrínseco": [
                f"${g.get('implied_price',0):,.2f}", f"${ex.get('implied_price',0):,.2f}"
            ],
            "Valor Empresa (EV)": [
                f"${g.get('enterprise_value',0):,.2f}", f"${ex.get('enterprise_value',0):,.2f}"
            ],
            "% en Valor Terminal": [
                f"{g.get('pct_from_tv',0)}%", f"{ex.get('pct_from_tv',0)}%"
            ],
            "Margen de Seguridad": [
                f"{max(0,(1-current_price/g.get('implied_price',1))*100):.1f}%" if g.get('implied_price',0) > 0 else "0%",
                f"{max(0,(1-current_price/ex.get('implied_price',1))*100):.1f}%" if ex.get('implied_price',0) > 0 else "0%",
            ],
        })
        pdf.data_table(dcf_df, col_widths=[55, 30, 35, 32, 32])

        # Projections table
        if dcf.get("projections"):
            pdf.subsection("Proyección de Flujos de Caja (FCF)")
            proj_df = pd.DataFrame(dcf["projections"])
            if not proj_df.empty:
                proj_df.columns = [c.replace("_", " ").title() for c in proj_df.columns]
                pdf.data_table(proj_df)

        # Scenario analysis
        if dcf_scenarios and "scenarios" in dcf_scenarios:
            pdf.subsection("Análisis de Escenarios (Bull / Base / Bear)")
            pdf.scenario_table(dcf_scenarios["scenarios"], current_price)

            # Sensitivity table
            sens = dcf_scenarios.get("sensitivity_wacc_tg")
            if sens:
                pdf.subsection("Tabla de Sensibilidad — WACC vs Crecimiento Terminal (Precio Intrínseco)")
                sens_df = pd.DataFrame(sens).T
                sens_df.index.name = "WACC \\ TG%"
                sens_df = sens_df.reset_index()
                pdf.data_table(sens_df, font_size=6.5)

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 6 — FINANCIAL STATEMENTS
    # ══════════════════════════════════════════════════════════════════════
    if statements and instrument != "ETF":
        pdf.add_page()
        pdf.section_title("6. Estados Financieros")

        stmt_map = {
            "income":   ("Estado de Resultados (Income Statement)", "income"),
            "balance":  ("Balance General (Balance Sheet)",         "balance"),
            "cashflow": ("Flujo de Caja (Cash Flow Statement)",     "cashflow"),
        }
        for key, (label, ai_key) in stmt_map.items():
            if key in statements and not statements[key].empty:
                pdf.subsection(label)
                pdf.data_table(statements[key].head(10), font_size=6.5)
                if ai_key in df_results and df_results[ai_key]:
                    pdf.body_text(str(df_results[ai_key])[:600], size=8)

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 7 — TECHNICAL ANALYSIS + CHARTS
    # ══════════════════════════════════════════════════════════════════════
    if results.get("technicals") or chart_paths:
        pdf.add_page()
        pdf.section_title("7. Análisis Técnico")

        if results.get("technicals"):
            pdf.body_text(results["technicals"])

        if chart_paths:
            captions = ["Gráfico Técnico con Indicadores (SMA, MACD, RSI, Bollinger)",
                        "Simulación Monte Carlo — Rutas de Precio",
                        "Análisis de Drawdown Histórico"]
            for path, cap in zip(chart_paths, captions):
                pdf.add_image_safe(path, w=170, caption=cap)

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 8 — RISK PROFILE
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("8. Perfil de Riesgo")

    if results.get("risk"):
        pdf.body_text(results["risk"])

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 9 — MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════
    if results.get("montecarlo"):
        pdf.section_title("9. Simulación Monte Carlo (GMM)")
        pdf.body_text(results["montecarlo"])

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 10 — OPTIONS (conditional)
    # ══════════════════════════════════════════════════════════════════════
    if results.get("options"):
        pdf.add_page()
        pdf.section_title("10. Análisis de Opciones")
        pdf.body_text(results["options"])

    # ETF-specific
    if results.get("etf"):
        pdf.add_page()
        pdf.section_title("Análisis Específico de ETF")
        pdf.body_text(results["etf"])

    # ══════════════════════════════════════════════════════════════════════
    #  PAGE 11 — INVESTMENT THESIS & FINAL RATING
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("11. Tesis de Inversión Final")

    if rating:
        pdf.rating_badge(rating, x=140, y=pdf.get_y() - 10)

    pdf.body_text(conclusion)

    # ══════════════════════════════════════════════════════════════════════
    #  LEGAL DISCLAIMER
    # ══════════════════════════════════════════════════════════════════════
    pdf.ln(10)
    pdf._rule(PDF_SLATE, 0.4)
    pdf.ln(4)
    y_disc = pdf.get_y()
    pdf.set_fill_color(*_LIGHT_GOLD)
    pdf.set_draw_color(*_BORDER)
    pdf.rect(10, y_disc, 190, 32, "FD")
    pdf.set_fill_color(*CCI_GOLD)
    pdf.rect(10, y_disc, 190, 3, "F")
    pdf.set_xy(14, y_disc + 6)
    pdf.set_font("Helvetica", "I", 7)
    pdf._set_color(PDF_SLATE, text=True)
    disclaimer = (
        "AVISO LEGAL: Este informe ha sido generado utilizando modelos de inteligencia artificial "
        "y datos de fuentes de acceso publico. No constituye asesoramiento financiero regulado ni "
        "una recomendacion de compra, venta o mantenimiento de ningun valor. Las estimaciones y "
        "proyecciones son de caracter ilustrativo. El rendimiento pasado no garantiza resultados futuros. "
        "CCI Puesto de Bolsa no asume responsabilidad por decisiones tomadas basandose en este documento. "
        "Consulte con un asesor financiero certificado antes de realizar inversiones. "
        "Confidencial - Uso exclusivo de clientes de CCI."
    )
    pdf.multi_cell(182, 4, pdf._sanitize(disclaimer))

    # ── Save ──
    filename = f"{ticker}_CCI_Research_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(filename)
    return filename


# ═══════════════════════════════════════════════════════════════════════════
#  Private helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_rating(text: str) -> str:
    """Attempt to extract investment rating from conclusion text."""
    if not text:
        return ""
    keywords = [
        "COMPRA FUERTE", "COMPRA FIRME",
        "COMPRA",
        "MANTENER", "MANTENER NEUTRAL",
        "VENTA FUERTE", "VENTA FIRME",
        "VENTA",
    ]
    text_upper = text.upper()
    for kw in keywords:
        if kw in text_upper:
            return kw
    return ""


def _extract_thesis_summary(text: str, max_chars: int = 450) -> str:
    """Extract first meaningful paragraph from the investment thesis."""
    if not text:
        return "Ver análisis detallado en la sección de Tesis de Inversión."
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("#")]
    summary = " ".join(lines[:5])
    return summary[:max_chars] + ("..." if len(summary) > max_chars else "")


def _add_two_col(pdf: PDFReport, left: str, right: str, title_l: str, title_r: str):
    """Helper to add two-column text block."""
    left_clean  = str(left or "No disponible.")[:700]
    right_clean = str(right or "No disponible.")[:700]
    pdf.two_col_text(left_clean, right_clean, title_l, title_r)
