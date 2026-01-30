import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from apps.track_record.lib.constants import STYLE_CSS_PATH, BG_JS_PATH, THEME_JS_PATH

def load_asset(path: Path) -> str:
    if not path.exists():
        return f"/* Asset not found: {path} */"
    return path.read_text(encoding="utf-8")

def inject_global_ui():
    """Injects CSS and JS for the custom 'AI Pro' design."""
    
    # 1. Inject CSS via st.markdown
    css_content = load_asset(STYLE_CSS_PATH)
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    
    # 2. Inject JS via st.components.v1.html
    # We wrap it in a hidden div or just a script tag.
    # Note: Streamlit components run in an iframe. To affect the parent, we need to be clever 
    # OR we accept that the background canvas lives in the iframe which is sized to full screen.
    # Standard Streamlit hack for full-page JS is usage of `window.parent` but that's brittle.
    # However, st.components.v1.html with height=0 can run JS.
    # BUT, to put a canvas BEHIND streamlit elements, we really want the JS to target window.parent.document.
    
    bg_js = load_asset(BG_JS_PATH)
    theme_js = load_asset(THEME_JS_PATH)
    
    # We modify the JS to execute on the PARENT window if possible, or we just render it.
    # Since Streamlit iframes are sandboxed, direct access to parent is often blocked on cloud.
    # BUT, for a local/docker app, we can try to style the iframe to be fixed full screen?
    # Actually, the best way for Streamlit is to inject the JS inside a script tag in markdown if allowed?
    # No, script tags are stripped from markdown.
    
    # Alternative: The "Aurora" background is rendered inside the Component Iframe, 
    # and we style that Iframe to be fixed position full screen behind everything.
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body, html {{ margin: 0; padding: 0; overflow: hidden; background: transparent; }}
    </style>
    </head>
    <body>
    <script>
        {bg_js}
        {theme_js}
    </script>
    </body>
    </html>
    """
    
    # This component will render the canvas. 
    # To make it a background, we rely on Streamlit's iframe container structure or we use a specific hack.
    # Hack: Make the iframe fixed position via `st.markdown` CSS targeting the iframe container.
    
    # Identify the iframe. It's hard to target a specific one.
    # A common trick is to render the component, and then use CSS to expand it.
    
    components.html(html_code, height=0)
    
    # CSS to make the specific iframe full screen fixed?
    # It is difficult to target the specific iframe created by components.html without an ID.
    # However, we can try to make the canvas simple and just let it sit at top?
    # For now, let's stick to a robust approach:
    # We inject the CSS that makes the main app transparent.
    # The `bg.js` tries to append to body. If it runs in iframe, it appends to iframe body.
    # So we need the iframe to be full screen background.
    
    st.markdown(
        """
        <style>
        /* Target the iframe container usually found at bottom or top */
        iframe[title="streamlit.components.v1.html"] {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
