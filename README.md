 RAG Local con Ollama + Gemma4
 
Sistema de Retrieval-Augmented Generation completamente local, construido con Ollama, Gemma4 y nomic-embed-text.
 
## Requisitos
 
- Python 3.10+
- [Ollama](https://ollama.com) instalado y corriendo
## Instalación
 
```bash
# 1. Clonar/descargar el proyecto
cd RAG
 
# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
 
# 3. Instalar dependencias
pip install ollama numpy pymupdf matplotlib seaborn rouge-score bert-score torch streamlit
 
# 4. Descargar modelos de Ollama
ollama pull gemma4
ollama pull nomic-embed-text
```
 
## Estructura del proyecto
 
```
RAG/
├── notebooks/
│   ├── rag.py           # Backend RAG (VectorDB + ask)
│   ├── app.py           # Interfaz web con Streamlit
│   └── analytics.py     # Métricas y gráficos de rendimiento
├── data/
│   └── docs/            # PDFs indexados
├── vector_db.json        # Índice vectorial (generado automáticamente)
└── README.md
```
 
## Uso
 
### Indexar documentos
 
Coloca tus PDFs en `data/docs/` y ejecuta:
 
```bash
python notebooks/rag.py
```
 
Solo es necesario hacerlo una vez. El índice se guarda en `vector_db.json` y se reutiliza automáticamente.
 
### Iniciar la interfaz web
 
```bash
streamlit run notebooks/app.py
```
 
Se abre en `http://localhost:8501`.
 
### Generar métricas y gráficos
 
```bash
python notebooks/analytics.py
```
 
Genera 6 gráficos PNG con métricas de precisión, calidad y velocidad.
 
---
 
## Arquitectura del Sistema
 
El pipeline RAG sigue un flujo de dos fases: **indexación** (offline, una sola vez) y **recuperación + generación** (online, por consulta).
 
### Flujo completo
 
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDFs en   │────▶│ Extracción  │────▶│  Chunking   │────▶│ Embeddings  │
│  data/docs/ │     │ (PyMuPDF)   │     │ CHUNK_SIZE  │     │  nomic-     │
└─────────────┘     └─────────────┘     │ CHUNK_OVERLAP│     │ embed-text  │
                                        └─────────────┘     └──────┬──────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Respuesta  │◀────│   Gemma4    │◀────│ Recuperación│◀────│  Vector DB  │
│  al usuario │     │  (Ollama)   │     │  TOP_K por  │     │ vector_db   │
└─────────────┘     └─────────────┘     │  similitud  │     │   .json     │
                                        └─────────────┘     └─────────────┘
```
 
### Descripción de cada etapa
 
1. **Ingestión con PyMuPDF** — Los archivos PDF se leen página a página usando la librería `pymupdf`. Se extrae el texto plano de cada página, eliminando encabezados, pies de página y artefactos de formato cuando es posible.
2. **División en chunks** — El texto extraído se fragmenta en bloques de `CHUNK_SIZE` caracteres (por defecto 800), con un solapamiento de `CHUNK_OVERLAP` caracteres (por defecto 150) entre fragmentos consecutivos. El solapamiento preserva el contexto en los límites de cada chunk.
3. **Generación de embeddings** — Cada chunk se convierte en un vector numérico de alta dimensión mediante el modelo `nomic-embed-text`, ejecutado localmente a través de Ollama. Este vector captura el significado semántico del fragmento.
4. **Almacenamiento en la base vectorial** — Los vectores se persisten en `vector_db.json` junto con el texto original y metadatos del documento de origen. El índice se carga en memoria en consultas posteriores.
5. **Recuperación por similitud semántica** — Ante una consulta del usuario, se genera su embedding y se compara contra todos los vectores del índice usando similitud coseno. Se seleccionan los `TOP_K` chunks más relevantes (por defecto 5).
6. **Generación de respuesta con Gemma4** — Los chunks recuperados se inyectan como contexto en el prompt enviado a `gemma4` a través de Ollama. El modelo genera una respuesta fundamentada exclusivamente en el contexto recuperado.
---
 
## Base Vectorial
 
El archivo `vector_db.json` actúa como índice vectorial local. Cada registro almacenado corresponde a un chunk indexado y contiene los siguientes campos:
 
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `text` | `string` | Texto original del fragmento tal como fue extraído del PDF |
| `embedding` | `float[]` | Vector numérico generado por `nomic-embed-text` (768 dimensiones) |
| `source` | `string` | Nombre del archivo PDF de origen |
| `chunk_id` | `int` | Índice ordinal del chunk dentro del documento fuente |
| `page` | `int` | Número de página aproximado del documento original |
 
El índice completo se carga en RAM en cada sesión. Para colecciones grandes (como el libro de Goodfellow o Russell & Norvig) el tiempo de carga inicial puede ser de varios segundos dependiendo del hardware disponible.
 
---
 
## Evaluación del Sistema
 
El módulo `analytics.py` implementa un conjunto de métricas que permiten evaluar la calidad del pipeline en sus dos dimensiones: recuperación y generación.
 
### Métricas de recuperación
 
- **Precision@K** — Proporción de chunks recuperados entre los `TOP_K` resultados que son efectivamente relevantes para la consulta. Evalúa la precisión del buscador vectorial antes de que el LLM intervenga.
### Métricas de calidad de generación
 
- **ROUGE Score** — Familia de métricas basadas en solapamiento de n-gramas entre la respuesta generada y una respuesta de referencia. Incluye ROUGE-1 (unigramas), ROUGE-2 (bigramas) y ROUGE-L (subsecuencia común más larga). Útil para medir cobertura léxica.
- **BERTScore** — Métrica de similitud semántica que compara las representaciones contextuales (embeddings BERT) de la respuesta generada y la referencia. Más robusta que ROUGE ante paráfrasis y sinónimos.
### Métricas de rendimiento
 
- **Tiempo de recuperación** — Latencia medida desde que se genera el embedding de la consulta hasta que se obtienen los `TOP_K` chunks del índice vectorial.
- **Tiempo de generación** — Latencia del modelo `gemma4` para producir la respuesta completa a partir del contexto recuperado.
Los 6 gráficos PNG que genera `analytics.py` presentan estas métricas de forma visual, facilitando la comparación entre distintas configuraciones de parámetros.
 
---
 
## Limitaciones actuales
 
El sistema es funcional para experimentación e investigación local, pero presenta las siguientes restricciones que conviene tener en cuenta:
 
- **Almacenamiento local plano** — La base vectorial se persiste como un único archivo JSON. No implementa estructuras de indexación aproximada (como HNSW o IVF), por lo que la búsqueda es lineal y escala de forma cuadrática con el número de chunks.
- **Reindexación completa** — Añadir o eliminar documentos requiere regenerar todo el índice desde cero. No existe actualización incremental del archivo `vector_db.json`.
- **Sin reranking avanzado** — Los chunks se devuelven ordenados únicamente por similitud coseno. No se aplican técnicas de reranking cruzado (cross-encoder) que podrían mejorar la relevancia de los resultados en consultas ambiguas.
- **Sin memoria conversacional** — Cada consulta es independiente. El sistema no mantiene el historial de la conversación, por lo que no es posible hacer preguntas de seguimiento que referencien respuestas anteriores.
- **Sensibilidad a los parámetros de chunking** — La calidad de las respuestas depende directamente de los valores de `CHUNK_SIZE`, `CHUNK_OVERLAP` y `TOP_K`. Valores inadecuados para un tipo de documento concreto pueden degradar significativamente los resultados sin que el sistema lo indique.
---
 
## Documentos indexados
 
| Documento | Chunks |
|-----------|--------|
| Attention Is All You Need (Vaswani et al., 2017) | ~61 |
| RAG for Knowledge-Intensive NLP Tasks (Lewis et al., 2020) | ~107 |
| REALM: Retrieval-Augmented LM Pre-Training (Guu et al., 2020) | ~78 |
| Deep Learning - Ian Goodfellow (2016) | ~2739 |
| Inteligencia Artificial: Un Enfoque Moderno - Russell & Norvig | ~5244 |
 
## Parámetros configurables
 
En `notebooks/rag.py`:
 
```python
EMBED_MODEL   = "nomic-embed-text"  # modelo de embeddings
CHAT_MODEL    = "gemma4"            # modelo generador
CHUNK_SIZE    = 800                 # caracteres por fragmento
CHUNK_OVERLAP = 150                 # solapamiento entre fragmentos
TOP_K         = 5                   # fragmentos recuperados por consulta
```