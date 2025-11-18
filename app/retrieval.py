# retrieval.py
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_pdf_text(path_pdf: str) -> str:
    lector = PdfReader(path_pdf)
    texto = []
    for pag in lector.pages:
        t = pag.extract_text() or ""
        if t.strip():
            texto.append(t.strip())
    return "\n\n".join(texto)


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """
    Trocea el texto en chunks de ~max_chars, respetando párrafos.
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.doc_mat = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_mat).ravel()
        idx = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in idx]
