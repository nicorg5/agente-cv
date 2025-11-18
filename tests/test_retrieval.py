# tests/test_retrieval.py
from app.retrieval import chunk_text, TfidfRetriever


def test_chunk_text_respeta_max_chars():
    """
    chunk_text debe devolver trozos que no excedan aproximadamente max_chars,
    salvo pequeños casos de bordes de párrafo.
    """
    texto = "Párrafo1. " * 50 + "\n\n" + "Párrafo2. " * 50
    max_chars = 200

    chunks = chunk_text(texto, max_chars=max_chars)

    assert len(chunks) >= 2  # Debería partirlo en varios trozos
    for ch in chunks:
        assert len(ch) <= max_chars + 20  # margen pequeño por seguridad


def test_tfidf_retriever_devuelve_chunks_relevantes():
    """
    Comprobamos que el retriever devuelve el chunk más relevante
    para una palabra clave evidente.
    """
    chunks = [
        "Me llamo Nicolás y trabajo con modelos de IA generativa.",
        "También tengo experiencia en análisis de datos y machine learning.",
        "En mi tiempo libre practico fútbol.",
    ]

    retriever = TfidfRetriever(chunks)

    # Pregunta muy orientada a deporte
    query = "¿Qué deportes practicas?"
    top = retriever.retrieve(query, k=1)

    assert len(top) == 1
    # Comprobación case-insensitive
    assert "fútbol" in top[0].lower()                  