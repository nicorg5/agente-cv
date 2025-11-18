# tests/test_backend.py
from app.backend import build_messages, ChatMessage
from app.utils import approx_tokens


def test_build_messages_incluye_system_y_user():
    """
    Verifica que build_messages genere al menos:
      - 1 mensaje de role="system"
      - 1 mensaje de role="user" con el contenido de la última pregunta.
    """
    mensaje = "¿Qué experiencia tienes en IA?"
    historial = []  # sin historial para el caso simple

    mensajes = build_messages(mensaje, historial)

    assert isinstance(mensajes, list)
    assert len(mensajes) >= 2

    # Debe haber un system al principio
    assert mensajes[0]["role"] == "system"
    assert "Resumen de" in mensajes[0]["content"] or "Actúas como" in mensajes[0]["content"]

    # Debe haberse incluido el mensaje del usuario
    assert any(
        m["role"] == "user" and "experiencia tienes en IA" in m["content"]
        for m in mensajes
    )


def test_build_messages_usa_historial_si_existe():
    """
    Si hay historial, esperamos que partes del historial se incorporen.
    No comprobamos el formato exacto, sólo que aparece contenido previo.
    """
    historial = [
        ChatMessage(role="user", content="Hola, cuéntame algo de tu perfil."),
        ChatMessage(role="assistant", content="Tengo experiencia en IA y datos."),
    ]
    mensaje = "¿Puedes profundizar en tu experiencia en datos?"

    mensajes = build_messages(mensaje, historial)

    # Debe aparecer el contenido del historial en algún sitio
    texto_completo = " ".join(m["content"] for m in mensajes)
    assert "cuéntame algo de tu perfil" in texto_completo
    assert "Tengo experiencia en IA y datos" in texto_completo
    assert "profundizar en tu experiencia en datos" in texto_completo

    # Y el tamaño en 'tokens' aproximados no debe ser 0 (sanity check)
    assert approx_tokens(texto_completo) > 0       