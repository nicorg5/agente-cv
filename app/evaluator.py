# app/evaluator.py
import json

from pydantic import BaseModel, ValidationError
from groq import Groq

from app.config import GROQ_API_KEY, EVAL_MODEL

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY no encontrada para evaluator. Revisa tu .env")

client_llama = Groq(api_key=GROQ_API_KEY)


class Evaluacion(BaseModel):
    es_aceptable: bool
    retroalimentacion: str


def build_system_prompt(nombre: str, resumen: str, perfil: str) -> str:
    return (
        f"Eres un evaluador que decide si una respuesta a una pregunta es aceptable.\n"
        f"Se te proporciona una conversación entre un Usuario y un AgenteIA que representa a {nombre}.\n"
        f"El Agente debe ser profesional y atractivo (potencial cliente o empleador).\n\n"
        f"Contexto de {nombre} (resumen y CV):\n\n"
        f"## Resumen:\n{resumen}\n\n## Perfil (CV):\n{perfil}\n\n"
        f"Instrucciones de salida: Responde EXCLUSIVAMENTE en JSON con la estructura exacta "
        f'{{"es_aceptable": true|false, "retroalimentacion": "texto"}}, sin texto adicional.'
    )


def format_history_for_eval(historial) -> str:
    """
    Acepta history en formato:
      - List[dict{role, content}] (como OpenAI/Gradio type='messages')
      - List[Tuple[user, assistant]]
    Devuelve un string legible.
    """
    if not historial:
        return "(sin historial previo)"

    lineas = []

    # caso dicts
    if isinstance(historial, list) and all(isinstance(x, dict) and "role" in x for x in historial):
        for x in historial:
            role = x.get("role")
            content = x.get("content")
            if not isinstance(content, str):
                continue
            if role == "user":
                lineas.append(f"Usuario: {content}")
            elif role == "assistant":
                lineas.append(f"AgenteIA: {content}")
        return "\n".join(lineas) if lineas else "(sin historial previo)"

    # caso tuplas
    for u, a in historial:
        if u:
            lineas.append(f"Usuario: {u}")
        if a:
            lineas.append(f"AgenteIA: {a}")

    return "\n".join(lineas) if lineas else "(sin historial previo)"


def build_user_prompt_for_eval(respuesta: str, mensaje: str, historial) -> str:
    hist_str = format_history_for_eval(historial)
    return (
        f"Conversación previa:\n{hist_str}\n\n"
        f"Último mensaje del Usuario:\n{mensaje}\n\n"
        f"Última respuesta del AgenteIA:\n{respuesta}\n\n"
        f"Evalúa si la respuesta es aceptable (contenido, tono, fidelidad al contexto).\n"
        f'Respóndeme SOLO con JSON válido: {{"es_aceptable": true|false, "retroalimentacion": "..."}}'
    )


def evaluar_respuesta(
    nombre: str,
    resumen: str,
    perfil: str,
    respuesta: str,
    mensaje: str,
    historial,
) -> Evaluacion:
    system_prompt = build_system_prompt(nombre, resumen, perfil)
    user_prompt = build_user_prompt_for_eval(respuesta, mensaje, historial)

    mensajes = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client_llama.chat.completions.create(
        model=EVAL_MODEL,
        messages=mensajes,
        response_format={"type": "json_object"},
        temperature=0,
    )
    contenido = resp.choices[0].message.content

    # Parse robusto
    try:
        return Evaluacion.model_validate_json(contenido)
    except ValidationError:
        data = json.loads(contenido)
        return Evaluacion.model_validate(data)