# app/backend.py
import time
import logging
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

from app.retrieval import read_pdf_text, chunk_text, TfidfRetriever
from app.utils import call_chat, budget_messages, should_evaluate
from app.evaluator import evaluar_respuesta
from app.config import GROQ_API_KEY, AGENT_MODEL, PDF_PATH, SUMMARY_PATH, NOMBRE

# ------------------------
# Configuración básica
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agente_cv_backend")

# (load_dotenv ya se hace en config, pero no molesta si se repite)
load_dotenv(override=True)

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY no encontrada. Revisa tu .env")

# ------------------------
# Inicialización de recursos globales
# ------------------------
logger.info("Cargando texto del CV desde %s", PDF_PATH)
texto_cv = read_pdf_text(PDF_PATH)
chunks_cv = chunk_text(texto_cv, max_chars=1200)
retriever = TfidfRetriever(chunks_cv)

logger.info("Leyendo resumen desde %s", SUMMARY_PATH)
with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    RESUMEN = f.read()

PERFIL = texto_cv  # texto completo del CV para el evaluador

client_openai = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Agente CV Backend", version="0.1.0")

# CORS (para que Gradio u otras UIs puedan llamar al backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción, restringe esto (dominio concreto)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Rate limiting ultra simple (en memoria)
# ------------------------
RATE_LIMIT_WINDOW = 60   # segundos
RATE_LIMIT_MAX = 20      # requests por ventana

# Estructura: {ip: [timestamps]}
_rate_limiter_store: dict[str, List[float]] = {}


def check_rate_limit(ip: str):
    now = time.time()
    lst = _rate_limiter_store.get(ip, [])
    # nos quedamos con las peticiones en la ventana
    lst = [t for t in lst if now - t < RATE_LIMIT_WINDOW]
    if len(lst) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    lst.append(now)
    _rate_limiter_store[ip] = lst


# ------------------------
# Modelos Pydantic para el API
# ------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    answer: str
    evaluated: bool
    es_aceptable: Optional[bool] = None
    retroalimentacion: Optional[str] = None


# ------------------------
# Helpers
# ------------------------
def history_to_messages(historial: Optional[List[ChatMessage]]) -> List[dict]:
    """
    Convierte la history del frontend (ChatMessage) a messages estilo OpenAI.
    Ignora roles no soportados.
    """
    if not historial:
        return []
    msgs: List[dict] = []
    for m in historial:
        if m.role in {"system", "user", "assistant"} and isinstance(m.content, str):
            msgs.append({"role": m.role, "content": m.content})
    return msgs


def build_messages(user_message: str, history: Optional[List[ChatMessage]]) -> List[dict]:
    """
    Construye la conversación completa para el LLM del agente:
    - System principal (persona de Nicolás + resumen)
    - Historial (user/assistant)
    - Contexto recuperado del CV (RAG ligero)
    - Mensaje de usuario
    """
    # 1) System base con resumen
    prompt_sistema = (
        f"Actúas como {NOMBRE} y respondes preguntas sobre su perfil profesional, "
        f"experiencia, habilidades y trayectoria. Si no sabes algo, dilo con honestidad.\n\n"
        f"## Resumen de {NOMBRE}:\n{RESUMEN[:12000]}\n"
    )
    mensajes: List[dict] = [{"role": "system", "content": prompt_sistema}]

    # 2) Historial formateado
    mensajes += history_to_messages(history)

    # 3) RAG: fragmentos del CV relevantes
    top_passages = retriever.retrieve(user_message, k=3)
    contexto = "\n\n".join(f"- {p}" for p in top_passages)
    mensajes.append(
        {
            "role": "system",
            "content": (
                f"## Fragmentos del CV relevantes:\n{contexto}\n\n"
                f"Usa estos fragmentos como evidencia principal; evita inventar datos no respaldados."
            ),
        }
    )

    # 4) input del usuario
    mensajes.append({"role": "user", "content": user_message})

    # 5) recorte de historial (tokens)
    mensajes = budget_messages(mensajes, max_tokens=6000)
    return mensajes


def reintentar_respuesta(
    respuesta: str,
    mensaje: str,
    history: Optional[List[ChatMessage]],
    retroalimentacion: str,
) -> str:
    """
    Construye un nuevo prompt de sistema explicando por qué la respuesta anterior fue rechazada
    y pide al modelo que la rehaga.
    """
    prompt_sistema_actualizado = (
        f"Actúas como {NOMBRE}. Tu respuesta anterior fue rechazada por el sistema de control de calidad.\n"
        f"Debes responder de nuevo corrigiendo lo siguiente:\n{retroalimentacion}\n\n"
        f"Evita repetir exactamente tu respuesta anterior. Mantén un tono profesional y fiel al perfil de {NOMBRE}."
    )

    mensajes: List[dict] = [{"role": "system", "content": prompt_sistema_actualizado}]
    mensajes += history_to_messages(history)
    mensajes.append({"role": "user", "content": mensaje})

    resp = call_chat(client_openai, AGENT_MODEL, mensajes)
    return resp.choices[0].message.content


# ------------------------
# Endpoint principal
# ------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, request: Request):
    # 1) rate limit
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(ip)

    user_msg = req.message
    history = req.history or []

    logger.info("Nueva petición de %s: %s", ip, user_msg)

    # 2) construir mensajes para el agente
    mensajes = build_messages(user_msg, history)

    # 3) llamada al modelo agente
    resp = call_chat(client_openai, AGENT_MODEL, mensajes)
    answer = resp.choices[0].message.content

    evaluated = False
    es_aceptable = None
    retroalimentacion = None

    # 4) decidir si evaluamos
    if should_evaluate(answer, user_msg):
        evaluated = True
        eval_res = evaluar_respuesta(
            nombre=NOMBRE,
            resumen=RESUMEN,
            perfil=PERFIL,
            respuesta=answer,
            mensaje=user_msg,
            historial=history_to_messages(history),
        )
        es_aceptable = eval_res.es_aceptable
        retroalimentacion = eval_res.retroalimentacion

        if not eval_res.es_aceptable:
            logger.info("Respuesta rechazada por el evaluador. Reintentando...")
            answer = reintentar_respuesta(answer, user_msg, history, eval_res.retroalimentacion)

    return ChatResponse(
        answer=answer,
        evaluated=evaluated,
        es_aceptable=es_aceptable,
        retroalimentacion=retroalimentacion,
    )

@app.get("/")
async def root():
    return {"status": "ok", "message": "Agente CV backend up"}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}