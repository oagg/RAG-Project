# analytics.py — Análisis completo de rendimiento del sistema RAG
#
# MÉTRICAS IMPLEMENTADAS:
#   1. Precisión de recuperación  — ¿trae los chunks correctos?
#   2. Calidad de respuesta       — ROUGE + BERTScore
#   3. Velocidad por etapa        — embedding, búsqueda, generación
#
# Requisitos:
#   pip install matplotlib seaborn numpy rouge-score bert-score torch
#
# Ejecutar: python analytics.py

import json
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import Counter

# Importar backend RAG 
sys.path.insert(0, str(Path(__file__).parent))
from rag import VectorDB, ask, get_embedding

#  CONFIGURACIÓN VISUAL
plt.rcParams.update({
    "figure.facecolor":  "#0e0e0f",
    "axes.facecolor":    "#141416",
    "axes.edgecolor":    "#2e2e33",
    "axes.labelcolor":   "#c4c2bc",
    "xtick.color":       "#888680",
    "ytick.color":       "#888680",
    "text.color":        "#e8e6e0",
    "grid.color":        "#2a2a2e",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "figure.dpi":        150,
})

ACCENT  = "#5b5bd6"
GREEN   = "#4ade80"
YELLOW  = "#facc15"
RED     = "#f87171"
BLUE    = "#60a5fa"
COLORS  = [ACCENT, GREEN, YELLOW, RED, BLUE]

NOMBRES_CORTOS = {
    "inteligencia-artificial-un-enfoque-moderno-stuart-j-russell.pdf": "Russell",
    "Deep+Learning+Ian+Goodfellow.pdf":                                 "Goodfellow",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": "RAG Paper",
    "REALM- Retrieval-Augmented Language Model Pre-Training.pdf":       "REALM",
    "Attention is all you need.pdf":                                    "Attention",
}


# ============================================================
#  DATASET DE EVALUACIÓN
#
#  Cada entrada tiene:
#    - question:       la pregunta de prueba
#    - expected_doc:   el documento donde DEBERÍA estar la respuesta
#    - reference:      respuesta de referencia para medir calidad
#
#  Estas respuestas de referencia están basadas en los papers/libros
#  y sirven como "gold standard" para ROUGE y BERTScore.
# ============================================================
EVAL_DATASET = [
    {
        "question": "¿Qué es el mecanismo de atención en Transformers?",
        "expected_doc": "Attention is all you need.pdf",
        "reference": (
            "El mecanismo de atención permite al modelo enfocarse en diferentes "
            "partes de la secuencia de entrada al generar cada token de salida. "
            "Calcula pesos de atención usando queries, keys y values, y usa "
            "multi-head attention para capturar distintos tipos de relaciones."
        ),
    },
    {
        "question": "¿Cómo funciona RAG para tareas intensivas en conocimiento?",
        "expected_doc": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf",
        "reference": (
            "RAG combina un recuperador denso que busca documentos relevantes "
            "con un generador seq2seq. El recuperador usa DPR para encontrar "
            "pasajes relevantes y el generador los usa como contexto adicional "
            "para producir respuestas más precisas y verificables."
        ),
    },
    {
        "question": "¿Qué es backpropagation y cómo entrena una red neuronal?",
        "expected_doc": "Deep+Learning+Ian+Goodfellow.pdf",
        "reference": (
            "Backpropagation es un algoritmo que calcula el gradiente de la "
            "función de pérdida respecto a los pesos de la red usando la regla "
            "de la cadena. Propaga el error desde la capa de salida hacia atrás, "
            "permitiendo actualizar los pesos para minimizar el error."
        ),
    },
    {
        "question": "¿Qué es un agente inteligente según Russell?",
        "expected_doc": "inteligencia-artificial-un-enfoque-moderno-stuart-j-russell.pdf",
        "reference": (
            "Un agente inteligente es cualquier entidad que percibe su entorno "
            "mediante sensores y actúa sobre él mediante actuadores. Un agente "
            "racional selecciona acciones que maximizan su medida de rendimiento "
            "basándose en la secuencia de percepciones y el conocimiento previo."
        ),
    },
    {
        "question": "¿Cómo pre-entrena REALM el recuperador de conocimiento?",
        "expected_doc": "REALM- Retrieval-Augmented Language Model Pre-Training.pdf",
        "reference": (
            "REALM pre-entrena el recuperador y el modelo de lenguaje de forma "
            "conjunta usando un objetivo de modelado del lenguaje enmascarado. "
            "El recuperador aprende a encontrar documentos útiles para predecir "
            "los tokens enmascarados, integrando conocimiento externo durante el pre-entrenamiento."
        ),
    },
]


#  MÉTRICAS AUXILIARES

def cosine_similarity(a, b):
    """Similitud coseno entre dos vectores."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L: mide la subsecuencia común más larga (LCS) entre
    la respuesta generada y la referencia. Valor entre 0 y 1.
    Más alto = más similar a la respuesta esperada.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(reference, hypothesis)
        return result["rougeL"].fmeasure
    except ImportError:
        print(" rouge-score no instalado: pip install rouge-score")
        return 0.0


def bertscore_f1(hypothesis: str, reference: str) -> float:
    """
    BERTScore: usa embeddings de BERT para comparar semánticamente
    la respuesta generada con la referencia. Más robusto que ROUGE
    porque captura similitud de significado, no solo palabras exactas.
    Valor entre 0 y 1. Más alto = respuesta semánticamente más cercana.
    """
    try:
        from bert_score import score as bscore
        P, R, F1 = bscore(
            [hypothesis], [reference],
            lang="es",
            verbose=False,
            device="cpu"   # usa CPU para no requerir GPU
        )
        return F1[0].item()
    except ImportError:
        print("  bert-score no instalado: pip install bert-score torch")
        return 0.0


# ============================================================
#  EVALUACIÓN 1 — PRECISIÓN DE RECUPERACIÓN
#
#  Mide si el documento correcto aparece entre los top-K
#  resultados para cada pregunta del dataset.
#
#  Hit@K = 1 si el doc esperado está en los K resultados
#  Hit@K = 0 si no está
#  Precisión@K = promedio de Hit@K sobre todas las preguntas
# ============================================================

def evaluar_recuperacion(db: VectorDB) -> dict:
    print("\n" + "="*50)
    print("MÉTRICA 1: Precisión de recuperación")
    print("="*50)

    resultados = []
    for item in EVAL_DATASET:
        retrieved = db.search(item["question"], top_k=5)
        docs_recuperados = [NOMBRES_CORTOS.get(r["source"], r["source"]) for r in retrieved]
        doc_esperado     = NOMBRES_CORTOS.get(item["expected_doc"], item["expected_doc"])
        hit              = doc_esperado in docs_recuperados
        top_score        = retrieved[0]["score"] if retrieved else 0.0

        resultados.append({
            "question":   item["question"][:50] + "...",
            "expected":   doc_esperado,
            "hit":        hit,
            "top_score":  top_score,
            "retrieved":  docs_recuperados,
        })

        estado = "Good" if hit else "No good"
        print(f"\n{estado} {item['question'][:60]}")
        print(f"   Esperado:    {doc_esperado}")
        print(f"   Recuperado:  {docs_recuperados[0]} (score={top_score:.3f})")

    precision = sum(r["hit"] for r in resultados) / len(resultados)
    print(f"\nPrecisión@5: {precision:.0%} ({sum(r['hit'] for r in resultados)}/{len(resultados)})")
    return {"resultados": resultados, "precision": precision}


# ============================================================
#  EVALUACIÓN 2 — CALIDAD DE RESPUESTA (ROUGE + BERTScore)
#
#  Para cada pregunta del dataset:
#    1. Genera una respuesta con el sistema RAG
#    2. La compara con la respuesta de referencia
#    3. Calcula ROUGE-L y BERTScore
# ============================================================

def evaluar_calidad(db: VectorDB) -> dict:
    print("\n" + "="*50)
    print("MÉTRICA 2: Calidad de respuesta")
    print("="*50)

    resultados = []
    for i, item in enumerate(EVAL_DATASET):
        print(f"\n[{i+1}/{len(EVAL_DATASET)}] {item['question'][:60]}...")

        resultado = ask(db, item["question"], verbose=False)
        respuesta = resultado["answer"]

        rouge  = rouge_l_score(respuesta, item["reference"])
        bert   = bertscore_f1(respuesta, item["reference"])

        resultados.append({
            "question": item["question"][:45] + "...",
            "rouge_l":  rouge,
            "bert_f1":  bert,
            "answer":   respuesta[:100],
        })

        print(f"   ROUGE-L:    {rouge:.3f}")
        print(f"   BERTScore:  {bert:.3f}")

    avg_rouge = np.mean([r["rouge_l"] for r in resultados])
    avg_bert  = np.mean([r["bert_f1"] for r in resultados])
    print(f"\nPromedio ROUGE-L:   {avg_rouge:.3f}")
    print(f"Promedio BERTScore: {avg_bert:.3f}")
    return {"resultados": resultados, "avg_rouge": avg_rouge, "avg_bert": avg_bert}


# ============================================================
#  EVALUACIÓN 3 — VELOCIDAD POR ETAPA
#
#  Mide el tiempo de cada paso del pipeline RAG:
#    1. Embedding de la pregunta
#    2. Búsqueda vectorial (similitud coseno)
#    3. Generación de respuesta (LLM)
#    4. Total end-to-end
#
#  Se repite N veces para obtener un promedio estable.
# ============================================================

def evaluar_velocidad(db: VectorDB, n_runs: int = 3) -> dict:
    print("\n" + "="*50)
    print("⚡ MÉTRICA 3: Velocidad por etapa")
    print("="*50)

    preguntas = [item["question"] for item in EVAL_DATASET[:3]]
    tiempos   = {"embedding": [], "busqueda": [], "generacion": [], "total": []}

    for pregunta in preguntas:
        for _ in range(n_runs):
            t0 = time.time()

            # Etapa 1: embedding de la pregunta
            t1 = time.time()
            q_emb = get_embedding(pregunta)
            t2 = time.time()

            # Etapa 2: búsqueda vectorial
            results = db.search(pregunta, top_k=5)
            t3 = time.time()

            # Etapa 3: generación
            resultado = ask(db, pregunta, verbose=False)
            t4 = time.time()

            tiempos["embedding"].append(t2 - t1)
            tiempos["busqueda"].append(t3 - t2)
            tiempos["generacion"].append(t4 - t3)
            tiempos["total"].append(t4 - t0)

            print(f"  embed={t2-t1:.2f}s  búsqueda={t3-t2:.3f}s  gen={t4-t3:.1f}s")

    promedios = {k: np.mean(v) for k, v in tiempos.items()}
    print(f"\nPromedios ({n_runs} runs × {len(preguntas)} preguntas):")
    for etapa, val in promedios.items():
        print(f"   {etapa:<12}: {val:.3f}s")

    return {"promedios": promedios, "tiempos": tiempos}

#  GRÁFICOS
def grafico_chunks_por_documento(chunks):
    conteo = Counter(NOMBRES_CORTOS.get(c["source"], c["source"]) for c in chunks)
    labels, valores = zip(*sorted(conteo.items(), key=lambda x: -x[1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, valores, color=COLORS[:len(labels)], width=0.55, zorder=2)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val:,}", ha="center", va="bottom", fontsize=10,
                color="#e8e6e0", fontweight="bold")
    ax.set_title("Fragmentos indexados por documento", fontsize=14, color="#f0ede6", pad=16)
    ax.set_ylabel("Número de chunks", fontsize=11)
    ax.set_ylim(0, max(valores) * 1.15)
    ax.grid(axis="y", zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("grafico_1_chunks.png", bbox_inches="tight")
    print("grafico_1_chunks.png")
    plt.show()


def grafico_precision_recuperacion(eval_rec: dict):
    resultados = eval_rec["resultados"]
    labels     = [r["question"][:30] + "..." for r in resultados]
    scores     = [r["top_score"] for r in resultados]
    colores    = [GREEN if r["hit"] else RED for r in resultados]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, scores, color=colores, height=0.55, zorder=2)

    # Línea de threshold
    ax.axvline(x=0.70, color=YELLOW, linestyle="--", linewidth=1.2, label="threshold 0.70")

    for bar, r in zip(bars, resultados):
        estado = "HIT" if r["hit"] else "MISS"
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                estado, va="center", fontsize=9,
                color=GREEN if r["hit"] else RED)

    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel("Score de recuperación (similitud coseno)", fontsize=11)
    ax.set_title(f"Precisión de recuperación  —  {eval_rec['precision']:.0%} Hit@5",
                 fontsize=14, color="#f0ede6", pad=16)
    ax.grid(axis="x", zorder=1)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("grafico_2_precision.png", bbox_inches="tight")
    print("grafico_2_precision.png")
    plt.show()


def grafico_calidad_respuesta(eval_cal: dict):
    resultados = eval_cal["resultados"]
    x          = np.arange(len(resultados))
    width      = 0.35
    labels     = [r["question"][:28] + "..." for r in resultados]
    rouge_vals = [r["rouge_l"] for r in resultados]
    bert_vals  = [r["bert_f1"] for r in resultados]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, rouge_vals, width, label="ROUGE-L",    color=ACCENT,  zorder=2)
    ax.bar(x + width/2, bert_vals,  width, label="BERTScore",  color=GREEN,   zorder=2)

    # Líneas de promedio
    ax.axhline(eval_cal["avg_rouge"], color=ACCENT, linestyle="--",
               linewidth=1, alpha=0.6, label=f"Avg ROUGE-L={eval_cal['avg_rouge']:.2f}")
    ax.axhline(eval_cal["avg_bert"],  color=GREEN,  linestyle="--",
               linewidth=1, alpha=0.6, label=f"Avg BERTScore={eval_cal['avg_bert']:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0–1)", fontsize=11)
    ax.set_title("Calidad de respuesta — ROUGE-L vs BERTScore", fontsize=14,
                 color="#f0ede6", pad=16)
    ax.legend(fontsize=9)
    ax.grid(axis="y", zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("grafico_3_calidad.png", bbox_inches="tight")
    print("grafico_3_calidad.png")
    plt.show()


def grafico_velocidad(eval_vel: dict):
    promedios = eval_vel["promedios"]
    etapas    = ["embedding", "busqueda", "generacion"]
    valores   = [promedios[e] for e in etapas]
    colores   = [BLUE, YELLOW, ACCENT]
    labels    = ["Embedding\n(pregunta→vector)", "Búsqueda\n(similitud coseno)", "Generación\n(LLM)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Barras de tiempo absoluto ──
    bars = ax1.bar(labels, valores, color=colores, width=0.5, zorder=2)
    for bar, val in zip(bars, valores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}s", ha="center", va="bottom", fontsize=11,
                 color="#e8e6e0", fontweight="bold")
    ax1.set_ylabel("Tiempo promedio (segundos)", fontsize=11)
    ax1.set_title("Tiempo por etapa del pipeline RAG", fontsize=13,
                  color="#f0ede6", pad=14)
    ax1.set_ylim(0, max(valores) * 1.25)
    ax1.grid(axis="y", zorder=1)
    ax1.spines[["top", "right", "left"]].set_visible(False)

    # ── Pie chart de proporción ──
    total = sum(valores)
    pcts  = [v / total * 100 for v in valores]
    wedges, texts, autotexts = ax2.pie(
        valores, labels=labels, colors=colores,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "#c4c2bc", "fontsize": 9},
        wedgeprops={"edgecolor": "#0e0e0f", "linewidth": 2}
    )
    for at in autotexts:
        at.set_color("#e8e6e0")
        at.set_fontweight("bold")
    ax2.set_title(f"Proporción del tiempo total ({total:.1f}s)",
                  fontsize=13, color="#f0ede6", pad=14)

    plt.tight_layout()
    plt.savefig("grafico_4_velocidad.png", bbox_inches="tight")
    print("grafico_4_velocidad.png")
    plt.show()


def grafico_similitud_documentos(chunks, embeddings):
    docs = {}
    for c, emb in zip(chunks, embeddings):
        label = NOMBRES_CORTOS.get(c["source"], c["source"])
        docs.setdefault(label, []).append(emb)

    doc_labels  = list(docs.keys())
    centroides  = np.array([np.mean(vecs, axis=0) for vecs in docs.values()])
    norms       = np.linalg.norm(centroides, axis=1, keepdims=True)
    c_norm      = centroides / norms
    sim_matrix  = c_norm @ c_norm.T

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(sim_matrix, xticklabels=doc_labels, yticklabels=doc_labels,
                annot=True, fmt=".2f",
                cmap=sns.color_palette("rocket_r", as_cmap=True),
                linewidths=0.5, linecolor="#2a2a2e",
                ax=ax, vmin=0.7, vmax=1.0)
    ax.set_title("Similitud coseno entre documentos\n(centroide de embeddings)",
                 fontsize=13, color="#f0ede6", pad=14)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("grafico_5_similitud.png", bbox_inches="tight")
    print("grafico_5_similitud.png")
    plt.show()


#  RESUMEN FINAL — dashboard de una sola imagen
def grafico_resumen(eval_rec, eval_cal, eval_vel):
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle("Resumen de rendimiento — RAG Local con Ollama + Gemma4",
                 fontsize=15, color="#f0ede6", y=1.01)

    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.4)

    # ── Panel 1: Precisión ──
    ax1 = fig.add_subplot(gs[0, 0])
    prec = eval_rec["precision"]
    ax1.pie([prec, 1 - prec], colors=[GREEN, "#2a2a2e"],
            startangle=90, wedgeprops={"edgecolor": "#0e0e0f", "linewidth": 2})
    ax1.text(0, 0, f"{prec:.0%}", ha="center", va="center",
             fontsize=22, color=GREEN, fontweight="bold")
    ax1.set_title("Precisión\nrecuperación@5", fontsize=10, color="#c4c2bc")

    # ── Panel 2: ROUGE-L ──
    ax2 = fig.add_subplot(gs[0, 1])
    rouge = eval_cal["avg_rouge"]
    ax2.pie([rouge, 1 - rouge], colors=[ACCENT, "#2a2a2e"],
            startangle=90, wedgeprops={"edgecolor": "#0e0e0f", "linewidth": 2})
    ax2.text(0, 0, f"{rouge:.2f}", ha="center", va="center",
             fontsize=22, color=ACCENT, fontweight="bold")
    ax2.set_title("ROUGE-L\npromedio", fontsize=10, color="#c4c2bc")

    # ── Panel 3: BERTScore ──
    ax3 = fig.add_subplot(gs[0, 2])
    bert = eval_cal["avg_bert"]
    ax3.pie([bert, 1 - bert], colors=[BLUE, "#2a2a2e"],
            startangle=90, wedgeprops={"edgecolor": "#0e0e0f", "linewidth": 2})
    ax3.text(0, 0, f"{bert:.2f}", ha="center", va="center",
             fontsize=22, color=BLUE, fontweight="bold")
    ax3.set_title("BERTScore F1\npromedio", fontsize=10, color="#c4c2bc")

    # ── Panel 4: Velocidad (ocupa toda la fila de abajo) ──
    ax4 = fig.add_subplot(gs[1, :])
    etapas  = ["Embedding", "Búsqueda vectorial", "Generación (LLM)"]
    tiempos = [eval_vel["promedios"]["embedding"],
               eval_vel["promedios"]["busqueda"],
               eval_vel["promedios"]["generacion"]]
    colores = [BLUE, YELLOW, ACCENT]
    bars    = ax4.barh(etapas, tiempos, color=colores, height=0.45, zorder=2)
    for bar, val in zip(bars, tiempos):
        ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}s", va="center", fontsize=10,
                 color="#e8e6e0", fontweight="bold")
    ax4.set_xlabel("Segundos", fontsize=10)
    ax4.set_title("Tiempo promedio por etapa", fontsize=11, color="#c4c2bc")
    ax4.grid(axis="x", zorder=1)
    ax4.spines[["top", "right", "left"]].set_visible(False)
    ax4.invert_yaxis()

    plt.savefig("grafico_0_resumen.png", bbox_inches="tight")
    print("grafico_0_resumen.png  ← este es el más importante para la presentación")
    plt.show()

if __name__ == "__main__":
    # Cargar DB 
    print("Cargando base de datos...")
    with open("vector_db.json") as f:
        raw = json.load(f)
    chunks     = raw["chunks"]
    embeddings = np.array(raw["embeddings"])

    db = VectorDB()
    db.load()
    print(f"{len(chunks)} fragmentos listos\n")

    # Correr evaluaciones
    eval_rec = evaluar_recuperacion(db)
    eval_cal = evaluar_calidad(db)       # tarda mas o menos 2 min (genera respuestas con el LLM)
    eval_vel = evaluar_velocidad(db, n_runs=2)

    # Generar gráficos 
    print("\nGenerando gráficos...\n")
    grafico_chunks_por_documento(chunks)
    grafico_precision_recuperacion(eval_rec)
    grafico_calidad_respuesta(eval_cal)
    grafico_velocidad(eval_vel)
    grafico_similitud_documentos(chunks, embeddings)
    grafico_resumen(eval_rec, eval_cal, eval_vel)

    print("\nListo — 6 PNGs guardados en la carpeta actual")