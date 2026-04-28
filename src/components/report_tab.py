# -*- coding: utf-8 -*-
"""Report tab — PDF generation, conclusion display, and download."""

import streamlit as st
import os


def render_report_tab(conclusion: str, pdf_path: str | None = None):
    """Render the Report tab."""

    # ── Conclusion ──
    st.markdown("#### 🎯 Conclusión de Inversión")

    if conclusion:
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,rgba(0,212,170,0.05),rgba(108,92,231,0.05));
                border:1px solid rgba(0,212,170,0.3);border-radius:12px;padding:25px;margin:10px 0;">
                {conclusion}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("Ejecuta el análisis completo para generar una conclusión de inversión.")

    st.divider()

    # ── PDF Download ──
    st.markdown("#### 📄 Reporte PDF")

    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
        st.download_button(
            label="📥 Descargar Reporte PDF",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
        st.success(f"✅ Reporte generado: **{os.path.basename(pdf_path)}**")
    else:
        st.info("El reporte PDF se generará automáticamente después de completar el análisis completo.")
