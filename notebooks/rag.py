# rag_ollama.py — Sistema RAG local con Ollama + Gemma
# Requisitos: pip install ollama numpy
# Modelos: ollama pull gemma3 && ollama pull nomic-embed-text

import ollama
import numpy as np
import json
import hashlib 
import fitz  
from pathlib import Path

# CONFIGURACIÓN
EMBED_MODEL  = "nomic-embed-text"   # modelo de embeddings
CHAT_MODEL   = "gemma4:latest"      # modelo de respuesta
CHUNK_SIZE   = 800                  # caracteres por fragmento
CHUNK_OVERLAP = 150                  # solapamiento entre fragmentos
TOP_K        = 5                    # cuántos fragmentos recuperar
DB_FILE      = "vector_db.json"     # archivo donde se guarda el índice


# UTILIDADES
 
def file_hash(path: str) -> str:
    """
    Calcula el hash MD5 de un archivo.
 
    Sirve para detectar si el contenido cambió desde la última
    vez que lo indexamos. Si el hash es el mismo → no re-indexar.
    Si cambió → eliminar los chunks viejos y volver a indexar.
    """
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
 
 
def read_file(path: str) -> str:
    """
    Lee el contenido de un archivo y lo devuelve como string.
 
    Soporta:
      - .txt y .md  → lectura directa con UTF-8
      - .pdf        → extracción de texto con pymupdf (fitz)
 
    Para añadir más formatos (docx, html, etc.) agrega
    un elif p.suffix.lower() == ".docx": aquí.
    """
    p = Path(path)
 
    if p.suffix.lower() == ".pdf":
        try:
            doc = fitz.open(path)
            # Extrae el texto de cada página y las une con saltos de línea
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            raise ImportError("Para leer PDFs instala pymupdf: pip install pymupdf")
    else:
        # Para .txt, .md y cualquier archivo de texto plano
        return p.read_text(encoding="utf-8")
 
 
def split_text(text: str, source: str = "") -> list[dict]:
    """
    Divide un texto largo en fragmentos (chunks) más pequeños.
 
    Por qué dividir:
      - Los modelos de embedding tienen un límite de tokens
      - Fragmentos pequeños permiten búsquedas más precisas
      - Con overlap evitamos perder información en los bordes
 
    Ejemplo visual con CHUNK_SIZE=20 y CHUNK_OVERLAP=5:
      Texto:   "AAAAABBBBBCCCCCDDDDDEEEEE"
      Chunk 1: "AAAAABBBBBCCCCCDDDDD"       (pos 0..20)
      Chunk 2: "CCCCCDDDDDEEEEE"            (pos 15..35)
                ^^^^^ estos 5 chars se repiten (overlap)
 
    Devuelve una lista de dicts: [{"text": "...", "source": "archivo.txt"}, ...]
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:  # ignorar chunks vacíos
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP  # avanzar dejando el overlap
    return chunks
 
 
def get_embedding(text: str) -> list[float]:
    """
    Convierte un texto en un vector numérico (embedding).
 
    El vector captura el "significado" semántico del texto.
    Textos con significados similares tendrán vectores cercanos
    en el espacio multidimensional.
 
    Ejemplo conceptual:
      "perro"  → [0.2, 0.8, 0.1, ...]
      "gato"   → [0.3, 0.7, 0.2, ...]   ← cercano a "perro"
      "avión"  → [0.9, 0.1, 0.6, ...]   ← lejos de ambos
 
    Usa nomic-embed-text que es rápido, local y muy preciso.
    """
    res = ollama.embed(model=EMBED_MODEL, input=text)
    return res.embeddings[0]  # devuelve el primer (y único) vector
 
 
# ============================================================
#  BASE DE DATOS VECTORIAL
#
#  Almacena los chunks y sus embeddings en memoria durante
#  la ejecución, y los persiste en un archivo JSON en disco.
#
#  PARA LA UI: los métodos devuelven/modifican estos atributos
#  que puedes exponer directamente:
#    - self.chunks          → lista de fragmentos indexados
#    - self.indexed_files   → diccionario de archivos procesados
# ============================================================
 
class VectorDB:
 
    def __init__(self):
        # Lista de fragmentos: [{"text": "...", "source": "archivo.txt"}, ...]
        self.chunks: list[dict] = []
 
        # Lista paralela de vectores: cada índice corresponde al chunk del mismo índice
        self.embeddings: list[list] = []
 
        # Registro de archivos indexados: {"/ruta/absoluta/archivo.txt": "hash_md5"}
        # Esto nos permite saber qué ya está indexado y si cambió
        self.indexed_files: dict[str, str] = {}
 
 
    # INDEXACIÓN 
 
    def add_document(self, path: str):
        """
        Indexa un archivo individual.
 
        Flujo:
          1. Convierte la ruta a absoluta (para evitar duplicados por ruta relativa)
          2. Calcula el hash del archivo
          3. Si ya está indexado y no cambió → lo salta
          4. Si cambió → elimina los chunks viejos y re-indexa
          5. Si es nuevo → lee, divide en chunks, genera embeddings y guarda
 
        PARA LA UI: puedes llamar este método desde un botón
        "Agregar documento" y luego llamar save() para persistir.
        """
        # Normalizar a ruta absoluta para evitar duplicados
        # ej: "doc.txt" y "./doc.txt" apuntan al mismo archivo
        p = str(Path(path).resolve())
        current_hash = file_hash(p)
 
        # Verificar si ya está en el índice
        if p in self.indexed_files:
            if self.indexed_files[p] == current_hash:
                # Mismo hash = mismo contenido = nada que hacer
                print(f"  ⏭  '{Path(p).name}' sin cambios, omitido")
                return
            else:
                # Hash diferente = el archivo fue modificado
                print(f"'{Path(p).name}' cambió, re-indexando...")
                self._remove_document(p)  # limpiar los chunks viejos primero
 
        # Leer y dividir el archivo en chunks
        print(f"Indexando '{Path(p).name}'...")
        text = read_file(p)
        new_chunks = split_text(text, source=Path(p).name)
 
        # Generar embedding para cada chunk y guardarlo
        for c in new_chunks:
            emb = get_embedding(c["text"])  # texto → vector
            self.chunks.append(c)           # guardar el chunk
            self.embeddings.append(emb)     # guardar su vector
 
        # Registrar el archivo con su hash actual
        self.indexed_files[p] = current_hash
        print(f"     → {len(new_chunks)} fragmentos agregados")
 
 
    def add_folder(self, folder: str, extensions: list[str] = None):
        """
        Indexa todos los archivos compatibles dentro de una carpeta.
 
        Busca recursivamente (incluye subcarpetas).
        Solo procesa archivos nuevos o modificados.
 
        PARA LA UI: puedes llamar este método desde un botón
        "Agregar carpeta" con un selector de directorio.
 
        Args:
            folder:     ruta a la carpeta
            extensions: lista de extensiones a incluir, ej: [".txt", ".pdf"]
                        por defecto procesa .txt, .md y .pdf
        """
        if extensions is None:
            extensions = [".txt", ".md", ".pdf"]
 
        folder_path = Path(folder)
        # rglob busca en todos los subdirectorios también
        files = [f for f in folder_path.rglob("*") if f.suffix.lower() in extensions]
 
        if not files:
            print(f"  No se encontraron archivos en '{folder}'")
            return
 
        print(f"\nIndexando carpeta '{folder}' ({len(files)} archivos)...")
        for f in files:
            self.add_document(str(f))
 
 
    def _remove_document(self, path: str):
        """
        Elimina todos los chunks de un archivo específico.
 
        Se usa internamente antes de re-indexar un archivo modificado.
        Filtra la lista de chunks conservando solo los que NO son del archivo.
 
        Args:
            path: ruta absoluta del archivo a eliminar del índice
        """
        source_name = Path(path).name
        # Índices de los chunks que queremos CONSERVAR (los de otros archivos)
        keep_idx = [i for i, c in enumerate(self.chunks) if c["source"] != source_name]
        # Reconstruir las listas solo con los chunks que se conservan
        self.chunks     = [self.chunks[i]     for i in keep_idx]
        self.embeddings = [self.embeddings[i] for i in keep_idx]
        # Eliminar del registro
        del self.indexed_files[path]
 
 
    # BÚSQUEDA 
 
    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Busca los chunks más relevantes para una pregunta.
 
        Usa similitud coseno: mide el ángulo entre dos vectores.
        Un ángulo de 0° (similitud=1.0) = idénticos en significado.
        Un ángulo de 90° (similitud=0.0) = completamente distintos.
 
        Matemáticamente:
          similitud = (A · B) / (|A| × |B|)
 
        Devuelve los top_k chunks con mayor similitud, ordenados
        de mayor a menor relevancia.
 
        PARA LA UI: puedes mostrar estos resultados como "fuentes"
        o "referencias" debajo de la respuesta del modelo.
 
        Returns:
            Lista de dicts con keys: text, source, score
        """
        if not self.embeddings:
            return []
 
        # Convertir la pregunta en vector
        q_emb  = np.array(get_embedding(query))
        # Matriz con todos los embeddings (un vector por fila)
        db_emb = np.array(self.embeddings)
 
        # Calcular similitud coseno entre la pregunta y TODOS los chunks
        # np.linalg.norm calcula la magnitud (longitud) de cada vector
        norms = np.linalg.norm(db_emb, axis=1) * np.linalg.norm(q_emb)
        # Producto punto dividido entre las magnitudes = similitud coseno
        sims  = db_emb @ q_emb / np.where(norms == 0, 1, norms)
 
        # Obtener los índices de los top_k chunks más similares
        # argsort ordena de menor a mayor, [::-1] invierte (mayor a menor)
        top_idx = np.argsort(sims)[::-1][:top_k]
 
        return [
            {**self.chunks[i], "score": float(sims[i])}
            for i in top_idx
        ]
 
 
    # PERSISTENCIA 
 
    def save(self, path: str = DB_FILE):
        """
        Guarda el índice completo en un archivo JSON.
 
        Guarda: chunks, embeddings y registro de archivos indexados.
        Llama a este método después de add_document() o add_folder()
        para no perder el trabajo al cerrar el programa.
 
        PARA LA UI: llama esto automáticamente después de indexar,
        o muestra un botón "Guardar índice".
        """
        data = {
            "chunks":        self.chunks,       # los fragmentos de texto
            "embeddings":    self.embeddings,   # sus vectores numéricos
            "indexed_files": self.indexed_files # registro de archivos y hashes
        }
        Path(path).write_text(json.dumps(data))
        print(f"\nDB guardada: {len(self.chunks)} fragmentos, {len(self.indexed_files)} archivos")
 
 
    def load(self, path: str = DB_FILE):
        """
        Carga el índice desde el archivo JSON.
 
        Llama esto al iniciar el programa para retomar el trabajo
        anterior sin necesidad de re-indexar todo.
 
        PARA LA UI: llama esto al arrancar la aplicación.
        """
        if not Path(path).exists():
            print("DB nueva — empezando desde cero")
            return
 
        data = json.loads(Path(path).read_text())
        self.chunks        = data.get("chunks", [])
        self.embeddings    = data.get("embeddings", [])
        self.indexed_files = data.get("indexed_files", {})
        print(f"DB cargada: {len(self.chunks)} fragmentos, {len(self.indexed_files)} archivos")
 
 
    def status(self) -> dict:
        """
        Devuelve el estado actual de la base de datos.
 
        PARA LA UI: usa el dict que devuelve para mostrar estadísticas
        en un panel lateral o dashboard (total de fragmentos, lista
        de archivos indexados, etc.)
 
        Returns:
            Dict con: total_chunks, total_files, files (lista de nombres)
        """
        info = {
            "total_chunks": len(self.chunks),
            "total_files":  len(self.indexed_files),
            "files":        [Path(p).name for p in self.indexed_files]
        }
        # También imprime en consola durante el desarrollo
        print(f"\nEstado de la DB:")
        print(f"   Fragmentos totales : {info['total_chunks']}")
        print(f"   Archivos indexados : {info['total_files']}")
        for name in info["files"]:
            print(f"     • {name}")
        return info
 
 
# ============================================================
#  MOTOR RAG
#  Combina búsqueda + generación en una sola función
# ============================================================
 
def ask(db: VectorDB, question: str, verbose: bool = False) -> dict:
    """
    Función principal del sistema RAG.
 
    Flujo:
      1. Busca los chunks más relevantes para la pregunta
      2. Construye un prompt con esos chunks como contexto
      3. Le pasa el prompt al LLM y devuelve la respuesta
 
    PARA LA UI: esta función devuelve un dict con todo lo necesario
    para renderizar la respuesta y sus fuentes en la interfaz:
      - answer:   el texto de respuesta del modelo
      - sources:  lista de fragmentos usados como referencia
                  (puedes mostrarlos como citas o tooltips)
      - question: la pregunta original (útil para el historial)
 
    Args:
        db:       instancia de VectorDB ya cargada
        question: pregunta del usuario en lenguaje natural
        verbose:  si True, imprime los chunks recuperados en consola
 
    Returns:
        {
          "answer":   "La respuesta del modelo...",
          "sources":  [{"text": "...", "source": "archivo.txt", "score": 0.87}, ...],
          "question": "¿La pregunta original?"
        }
    """
    # Paso 1: recuperar los chunks más relevantes
    results = db.search(question)
 
    if not results:
        return {
            "answer":   "No hay documentos indexados todavía.",
            "sources":  [],
            "question": question
        }
 
    # Paso 2: construir el bloque de contexto para el prompt
    # Cada chunk se etiqueta con su archivo de origen
    context = "\n\n---\n\n".join(
        f"[{r['source']}]\n{r['text']}" for r in results
    )
 
    # Log de debug (útil durante el desarrollo)
    if verbose:
        print("\nFragmentos recuperados:")
        for r in results:
            print(f"score={r['score']:.3f} | {r['source']} | {r['text'][:80]}...")
 
    # Paso 3: construir el prompt RAG
    # La clave es restringir al modelo a usar SOLO el contexto provisto,
    # así evitamos alucinaciones basadas en su memoria de entrenamiento
    prompt = f"""Eres un asistente académico experto. 
    Responde la pregunta basándote en el siguiente contexto.
    Sé detallado y explica con tus propias palabras usando la información del contexto.

    CONTEXTO: {context}

    PREGUNTA: {question}
    RESPUESTA:"""
    
   
    # Paso 4: llamar al LLM con el prompt enriquecido
    res = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
 
    # Devolver respuesta + fuentes en un dict estructurado
    # (listo para ser consumido por una UI)
    return {
        "answer":   res.message.content,
        "sources":  results,   # los chunks usados como contexto
        "question": question
    }
 
 
#  EJEMPLO DE USO
 
if __name__ == "__main__":
 
    # Inicializar la DB y cargar el índice existente (si hay)
    db = VectorDB()
    db.load()  # carga lo que ya estaba indexado

    # Preguntas de prueba
    resultado = ask(db, "¿Qué es un agente inteligente?", verbose=True)
    print(resultado["answer"])


    #db.add_folder("data/docs/", extensions=[".pdf"])
    #db.save()
    #db.status()
