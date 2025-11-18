# utils.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIStatusError


def approx_tokens(s: str) -> int:
    """
    Aproximación muy simple: 1 token ≈ 4 caracteres.
    No es exacto pero es suficiente para cortar historial.
    """
    return max(1, len(s) // 4)


def budget_messages(messages: list[dict], max_tokens: int = 6000) -> list[dict]:
    """
    Mantiene el primer/primeros mensajes de system y recorta del historial lo más viejo
    hasta que el total estimado de tokens sea <= max_tokens.
    """
    if not messages:
        return messages

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    other = [m for m in messages if m.get("role") != "system"]

    total = sum(approx_tokens(m.get("content", "")) for m in sys_msgs)
    kept = []

    for m in reversed(other):
        t = approx_tokens(m.get("content", ""))
        if total + t > max_tokens:
            break
        kept.append(m)
        total += t

    return sys_msgs + list(reversed(kept))


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIStatusError)
    ),
)
def call_chat(client, model: str, messages: list[dict], **kwargs):
    """
    Envoltura con reintentos para client.chat.completions.create(...)
    """
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


def should_evaluate(answer: str, user_msg: str) -> bool:
    """
    Heurística para decidir si merece la pena evaluar (y pagar el evaluador).
    """
    if approx_tokens(answer) > 150:  # ~600+ tokens aprox
        return True

    keywords = [
        "experiencia",
        "años",
        "responsable de",
        "certificado",
        "título",
        "salario",
        "senior",
        "lead",
    ]
    if any(k.lower() in user_msg.lower() for k in keywords):
        return True

    return False
