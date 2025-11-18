# frontend_gradio.py
import httpx
import gradio as gr


BACKEND_URL = "http://127.0.0.1:8000/chat"


def gradio_chat(message, history):
    """
    Gradio pasa:
      - message: str
      - history: List[Dict{role, content}]  (por type="messages")
    Enviamos message + history al backend y devolvemos sólo 'answer'.
    """
    payload = {
        "message": message,
        "history": history or [],
    }

    # Llamada síncrona al backend
    try:
        resp = httpx.post(BACKEND_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error al contactar con el backend: {e}"

    data = resp.json()
    answer = data.get("answer", "")

    # Opcional: podríamos mostrar retroalimentación si evaluated == True
    # pero por ahora devolvemos solo la respuesta del agente.
    return answer


demo = gr.ChatInterface(
    fn=gradio_chat,
    type="messages",
    title="Asistente de CV de Nicolás",
    description="Hazme una pregunta sobre mi perfil profesional",
)

if __name__ == "__main__":
    demo.launch()
