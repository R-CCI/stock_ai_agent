# -*- coding: utf-8 -*-
"""Google OAuth2 Authentication for Stock AI Agent.

Implements Google Sign-In with domain restriction to @cci.com.do.

Usage in app.py:
    from src.auth import require_auth, render_logout_button
    if not require_auth():
        st.stop()
"""

import streamlit as st
import urllib.parse
import secrets
import requests
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_DOMAIN = "cci.com.do"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def _get_oauth_config() -> dict:
    """Read OAuth credentials from Streamlit secrets.

    Supports two formats:
      [oauth] section (preferred):
          [oauth]
          client_id     = "..."
          client_secret = "..."
          redirect_uri  = "..."

      Top-level keys (legacy):
          GOOGLE_CLIENT_ID     = "..."
          GOOGLE_CLIENT_SECRET = "..."
          GOOGLE_REDIRECT_URI  = "..."
    """
    try:
        # Prefer [oauth] section
        if "oauth" in st.secrets:
            sec = st.secrets["oauth"]
            return {
                "client_id":     sec.get("client_id", ""),
                "client_secret": sec.get("client_secret", ""),
                "redirect_uri":  sec.get("redirect_uri", "http://localhost:8501"),
            }
        # Fall back to top-level keys
        return {
            "client_id":     st.secrets.get("GOOGLE_CLIENT_ID", ""),
            "client_secret": st.secrets.get("GOOGLE_CLIENT_SECRET", ""),
            "redirect_uri":  st.secrets.get("GOOGLE_REDIRECT_URI", "http://localhost:8501"),
        }
    except Exception:
        return {}


def _build_auth_url(config: dict, state: str) -> str:
    """Build the Google OAuth2 authorization URL."""
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def _exchange_code_for_token(code: str, config: dict) -> Optional[dict]:
    """Exchange authorization code for access + id tokens."""
    try:
        resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "redirect_uri": config["redirect_uri"],
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error al intercambiar código OAuth: {e}")
        return None


def _get_user_info(access_token: str) -> Optional[dict]:
    """Fetch user profile from Google."""
    try:
        resp = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _validate_domain(email: str) -> bool:
    """Verify the email belongs to the allowed corporate domain."""
    return email.lower().strip().endswith(f"@{ALLOWED_DOMAIN}")


def _render_login_page(auth_url: str):
    """Render a professional login page."""
    st.markdown(
        """
        <style>
            .login-container {
                max-width: 420px;
                margin: 80px auto;
                padding: 48px 40px;
                background: linear-gradient(145deg, #1a1f2e, #161b27);
                border: 1px solid rgba(0,212,170,0.2);
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            }
            .login-logo {
                font-size: 3.5rem;
                margin-bottom: 16px;
            }
            .login-title {
                font-size: 1.8rem;
                font-weight: 800;
                background: linear-gradient(135deg, #00D4AA, #6C5CE7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
            }
            .login-subtitle {
                color: #8892a4;
                font-size: 0.9rem;
                margin-bottom: 32px;
                line-height: 1.5;
            }
            .login-btn {
                display: inline-flex;
                align-items: center;
                gap: 12px;
                background: white;
                color: #1a1f2e !important;
                padding: 14px 28px;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                text-decoration: none !important;
                transition: all 0.2s;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .login-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .login-divider {
                color: #4a5568;
                margin: 24px 0;
                font-size: 0.8rem;
            }
            .login-domain-note {
                color: #00D4AA;
                font-size: 0.82rem;
                margin-top: 20px;
                padding: 8px 16px;
                background: rgba(0,212,170,0.08);
                border-radius: 8px;
                border: 1px solid rgba(0,212,170,0.2);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="login-container">
            <div class="login-logo">📊</div>
            <div class="login-title">Stock AI Agent</div>
            <div class="login-subtitle">
                Análisis de Inversión de Grado Institucional<br>
                potenciado por Inteligencia Artificial
            </div>
            <a href="{auth_url}" class="login-btn" target="_self">
                <svg width="20" height="20" viewBox="0 0 48 48">
                    <path fill="#EA4335" d="M24 9.5c3.5 0 6.4 1.2 8.8 3.2l6.5-6.5C35.2 2.9 29.9.5 24 .5 14.7.5 6.9 6.1 3.5 14L11 19.8C12.8 13.8 17.9 9.5 24 9.5z"/>
                    <path fill="#4285F4" d="M46.5 24.5c0-1.5-.1-3-.4-4.5H24v8.5h12.7c-.6 3-2.3 5.6-4.9 7.3l7.7 6c4.5-4.2 7-10.3 7-17.3z"/>
                    <path fill="#FBBC05" d="M11 28.2C10.5 26.6 10.2 24.8 10.2 23s.3-3.6.8-5.2L3.5 12C1.3 16.2 0 20.9 0 26s1.3 9.8 3.5 14l7.5-5.8z"/>
                    <path fill="#34A853" d="M24 47.5c5.9 0 11.2-2 15-5.4l-7.7-6c-2 1.4-4.6 2.2-7.3 2.2-6.1 0-11.2-4.3-13-10.3L3.5 34C6.9 41.9 14.7 47.5 24 47.5z"/>
                </svg>
                Continuar con Google
            </a>
            <div class="login-domain-note">
                🔒 Acceso restringido a cuentas @cci.com.do
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_logout_button():
    """Render logout button in the sidebar."""
    user = st.session_state.get("auth_user", {})
    email = user.get("email", "")
    name = user.get("name", "Usuario")
    picture = user.get("picture", "")

    with st.sidebar:
        st.markdown("---")
        cols = st.columns([1, 3])
        with cols[0]:
            if picture:
                st.image(picture, width=36)
            else:
                st.markdown("👤")
        with cols[1]:
            st.markdown(f"**{name}**")
            st.caption(email)

        if st.button("🚪 Cerrar Sesión", use_container_width=True, key="logout_btn"):
            for key in ["auth_user", "auth_state", "auth_code_used"]:
                st.session_state.pop(key, None)
            st.rerun()


def _render_dev_mode_gate():
    """Show a dev-mode warning page when OAuth is not configured."""
    st.markdown(
        """
        <style>
            .dev-container {
                max-width: 460px;
                margin: 80px auto;
                padding: 48px 40px;
                background: linear-gradient(145deg, #1a1f2e, #161b27);
                border: 1px solid rgba(255,193,7,0.35);
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            }
            .dev-logo { font-size: 3rem; margin-bottom: 16px; }
            .dev-title {
                font-size: 1.7rem;
                font-weight: 800;
                background: linear-gradient(135deg, #00D4AA, #6C5CE7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }
            .dev-badge {
                display: inline-block;
                background: rgba(255,193,7,0.15);
                border: 1px solid rgba(255,193,7,0.5);
                color: #FFC107;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 1px;
                padding: 4px 14px;
                border-radius: 20px;
                margin-bottom: 20px;
            }
            .dev-note {
                color: #8892a4;
                font-size: 0.85rem;
                line-height: 1.6;
                margin-bottom: 28px;
            }
            .dev-setup {
                background: rgba(0,212,170,0.07);
                border: 1px solid rgba(0,212,170,0.2);
                border-radius: 10px;
                padding: 14px 18px;
                text-align: left;
                color: #a0b0c0;
                font-size: 0.78rem;
                line-height: 1.7;
                margin-bottom: 24px;
            }
            .dev-setup code {
                background: rgba(255,255,255,0.08);
                padding: 1px 5px;
                border-radius: 4px;
                color: #00D4AA;
                font-size: 0.78rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dev-container">
            <div class="dev-logo">📊</div>
            <div class="dev-title">Stock AI Agent</div>
            <div class="dev-badge">MODO DESARROLLO</div>
            <div class="dev-note">
                Google OAuth2 no esta configurado.<br>
                Para habilitar autenticacion, crea
                <code>.streamlit/secrets.toml</code> con tus credenciales.
            </div>
            <div class="dev-setup">
                <strong style="color:#00D4AA;">secrets.toml requerido:</strong><br>
                <code>GOOGLE_CLIENT_ID = "..."</code><br>
                <code>GOOGLE_CLIENT_SECRET = "..."</code><br>
                <code>GOOGLE_REDIRECT_URI = "http://localhost:8501"</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "Continuar en Modo Desarrollo",
            use_container_width=True,
            type="primary",
            key="dev_mode_btn",
        ):
            st.session_state["dev_mode_accepted"] = True
            st.rerun()


def require_auth() -> bool:
    """
    Main auth gate. Returns True if the user is authenticated.

    Call this at the top of main() before rendering anything else:
        if not require_auth():
            st.stop()
    """
    config = _get_oauth_config()

    # If OAuth is not configured, show dev-mode gate instead of silent bypass
    if not config.get("client_id") or not config.get("client_secret"):
        if st.session_state.get("dev_mode_accepted"):
            return True
        _render_dev_mode_gate()
        return False

    # Already authenticated
    if st.session_state.get("auth_user"):
        return True

    # ── Handle OAuth callback ──────────────────────────────────────────────
    query_params = st.query_params
    code = query_params.get("code")
    state = query_params.get("state")

    if code and not st.session_state.get("auth_code_used"):
        # Validate CSRF state
        stored_state = st.session_state.get("auth_state")
        if stored_state and state != stored_state:
            st.error("⚠️ Estado OAuth inválido. Intenta de nuevo.")
            st.session_state.pop("auth_state", None)
            st.rerun()
            return False

        # Mark code as used to prevent double-processing
        st.session_state["auth_code_used"] = True

        with st.spinner("Verificando credenciales..."):
            tokens = _exchange_code_for_token(code, config)
            if not tokens:
                st.error("Error al obtener tokens. Intenta de nuevo.")
                return False

            user_info = _get_user_info(tokens.get("access_token", ""))
            if not user_info:
                st.error("Error al obtener información del usuario.")
                return False

            email = user_info.get("email", "")

            # Domain validation
            if not _validate_domain(email):
                st.error(
                    f"❌ **Acceso Denegado**\n\n"
                    f"La cuenta `{email}` no pertenece al dominio `@{ALLOWED_DOMAIN}`.\n\n"
                    f"Solo se permiten cuentas corporativas de CCI."
                )
                return False

            # Store user in session
            st.session_state["auth_user"] = {
                "email": email,
                "name": user_info.get("name", ""),
                "picture": user_info.get("picture", ""),
                "sub": user_info.get("sub", ""),
            }

            # Clean up URL params
            st.query_params.clear()
            st.rerun()
            return True

    # ── Not authenticated — show login page ───────────────────────────────
    state_token = secrets.token_urlsafe(32)
    st.session_state["auth_state"] = state_token
    auth_url = _build_auth_url(config, state_token)

    _render_login_page(auth_url)
    return False
