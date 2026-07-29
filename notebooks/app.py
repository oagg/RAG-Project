# app.py — Interfaz web del sistema RAG
# Ejecutar con: streamlit run app.py

import streamlit as st
import sys
from pathlib import Path

# Importar el backend RAG (debe estar en la misma carpeta)
from rag import VectorDB, ask

#  CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="RAG Local",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  ESTILOS PERSONALIZADOS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Fondo oscuro principal */
.stApp {
    background-color: #0e0e0f;
    color: #e8e6e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #141416;
    border-right: 1px solid #2a2a2e;
}
[data-testid="stSidebar"] * {
    color: #c4c2bc !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background-color: #1c1c1f !important;
    border: 1px solid #2e2e33 !important;
    border-radius: 8px !important;
    color: #e8e6e0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #5b5bd6 !important;
    box-shadow: 0 0 0 2px rgba(91,91,214,0.2) !important;
}

/* ── Botones ── */
.stButton > button {
    background-color: #5b5bd6 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background-color: #6e6ede !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(91,91,214,0.3) !important;
}

/* ── Burbujas de chat ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0;
}
.msg-user .bubble {
    background: #5b5bd6;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 70%;
    font-size: 15px;
    line-height: 1.6;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 16px 0;
    gap: 12px;
    align-items: flex-start;
}
.bot-avatar {
    width: 32px;
    height: 32px;
    background: #1c1c1f;
    border: 1px solid #2e2e33;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-bot .bubble {
    background: #1c1c1f;
    color: #e8e6e0;
    border: 1px solid #2e2e33;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 75%;
    font-size: 15px;
    line-height: 1.7;
}

/* ── Fuentes / Sources ── */
.sources-container {
    margin-top: 10px;
    max-width: 75%;
    margin-left: 44px;
}
.source-tag {
    display: inline-block;
    background: #0e0e0f;
    border: 1px solid #2e2e33;
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #888680;
    margin: 3px 3px 0 0;
    cursor: default;
}
.source-tag:hover {
    border-color: #5b5bd6;
    color: #a8a6ff;
}
.score-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}

/* ── Encabezado ── */
.rag-header {
    padding: 32px 0 24px 0;
    border-bottom: 1px solid #2a2a2e;
    margin-bottom: 32px;
}
.rag-title {
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #f0ede6;
    margin: 0;
}
.rag-subtitle {
    font-size: 13px;
    color: #5c5a55;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Stats badges ── */
.stat-badge {
    display: inline-block;
    background: #1c1c1f;
    border: 1px solid #2e2e33;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    color: #888680;
    margin-right: 8px;
}
.stat-badge span {
    color: #a8a6ff;
    font-weight: 500;
}

/* ── File list en sidebar ── */
.file-item {
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    color: #888680;
    border: 1px solid #2a2a2e;
    margin-bottom: 6px;
    background: #0e0e0f;
    word-break: break-all;
}

/* ── Área de chat ── */
.chat-area {
    min-height: 400px;
    padding-bottom: 20px;
}

/* Ocultar elementos de Streamlit que no queremos */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 860px; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0e0e0f; }
::-webkit-scrollbar-thumb { background: #2e2e33; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


#  INICIALIZAR BASE DE DATOS (solo una vez por sesión)
@st.cache_resource
def load_db():
    """
    Carga la VectorDB una sola vez y la mantiene en memoria.
    @st.cache_resource evita recargarla en cada interacción.
    """
    db = VectorDB()
    db.load()
    return db

db = load_db()


#  ESTADO DE SESIÓN
if "messages" not in st.session_state:
    st.session_state.messages = []   # historial del chat
if "indexing" not in st.session_state:
    st.session_state.indexing = False


#  SIDEBAR — Panel de documentos
with st.sidebar:
    st.markdown("### Documentos")

    #  Subir archivos 
    uploaded = st.file_uploader(
        "Agregar PDFs o TXTs",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded:
        if st.button("Indexar archivos"):
            docs_dir = Path("data/docs")
            docs_dir.mkdir(parents=True, exist_ok=True)

            progress = st.progress(0, text="Indexando...")
            for i, f in enumerate(uploaded):
                dest = docs_dir / f.name
                dest.write_bytes(f.getbuffer())
                db.add_document(str(dest))
                progress.progress((i + 1) / len(uploaded), text=f"Indexando {f.name}...")

            db.save()
            progress.empty()
            st.success(f"✓ {len(uploaded)} archivo(s) indexados")
            st.cache_resource.clear()
            st.rerun()

    st.divider()

    #Lista de archivos indexados 
    status = db.status()

    if status["total_files"] == 0:
        st.caption("No hay documentos indexados aún.")
    else:
        st.caption(f"{status['total_files']} archivo(s) · {status['total_chunks']} fragmentos")
        for fname in status["files"]:
            st.markdown(f'<div class="file-item"> {fname}</div>', unsafe_allow_html=True)

    st.divider()

    # Botón para limpiar chat 
    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

    # Opción verbose 
    verbose = st.toggle("Mostrar fuentes detalladas", value=True)


#  CONTENIDO PRINCIPAL

# Header
st.markdown("""
<div class="rag-header">
    <p class="rag-title">RAG Local</p>
    <p class="rag-subtitle">retrieval-augmented generation · ollama · gemma4</p>
</div>
""", unsafe_allow_html=True)

# Stats rápidas
if db.chunks:
    st.markdown(f"""
    <div style="margin-bottom: 28px;">
        <span class="stat-badge">fragmentos <span>{len(db.chunks)}</span></span>
        <span class="stat-badge">documentos <span>{len(db.indexed_files)}</span></span>
        <span class="stat-badge">modelo <span>gemma4</span></span>
    </div>
    """, unsafe_allow_html=True)

# Historial del chat
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 60px 0; color: #3a3a3f;">
            <div style="font-size: 40px; margin-bottom: 16px;">🔍</div>
            <div style="font-size: 15px; font-weight: 500; color: #5c5a55;">Haz una pregunta sobre tus documentos</div>
            <div style="font-size: 13px; color: #3a3a3f; margin-top: 6px; font-family: monospace;">
                Los documentos indexados aparecen en el panel izquierdo
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-bot">
                <div class="bot-avatar">◈</div>
                <div class="bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Mostrar fuentes si existen
            if verbose and msg.get("sources"):
                tags = ""
                for s in msg["sources"]:
                    score = s["score"]
                    # Color del dot según relevancia
                    if score > 0.75:
                        color = "#4ade80"   # verde = muy relevante
                    elif score > 0.5:
                        color = "#facc15"  # amarillo = relevante
                    else:
                        color = "#f87171"  # rojo = poco relevante
                    tags += f'<span class="source-tag"><span class="score-dot" style="background:{color}"></span>{s["source"]} {score:.2f}</span>'

                st.markdown(f'<div class="sources-container">{tags}</div>', unsafe_allow_html=True)


#  Input de pregunta
st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Pregunta",
            placeholder="¿Cómo funciona el mecanismo de atención en Transformers?",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Enviar")

# Procesar pregunta
if submitted and question.strip():
    if not db.chunks:
        st.warning("Primero indexa al menos un documento desde el panel izquierdo.")
    else:
        # Agregar pregunta del usuario al historial
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Generar respuesta con spinner
        with st.spinner("Buscando en documentos..."):
            resultado = ask(db, question, verbose=False)

        # Agregar respuesta al historial
        st.session_state.messages.append({
            "role": "assistant",
            "content": resultado["answer"],
            "sources": resultado["sources"]
        })

        st.rerun()