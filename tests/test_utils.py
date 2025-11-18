import pytest

from app.utils import approx_tokens, budget_messages, should_evaluate

def test_approx_tokens_crece_con_longitud():
    short = "hola"
    long = "hola " * 100

    assert approx_tokens(short) < approx_tokens(long)
    assert approx_tokens(short) >= 1  # nunca 0

def test_budget_messages_recorta_historial():
    system_msg = {"role": "system", "content": "System prompt"} 
    # Simulamos 10 mensajes largos de user/assistant
    long_content = "x" * 2000
    other_msgs = [
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": long_content},
    ] * 10

    mensajes = [system_msg] + other_msgs

    recortados = budget_messages(mensajes, max_tokens=3000)

    # System debe mantenerse siempre
    assert recortados[0]["role"] == "system"
    # El total de mensajes recortados debe ser menor al original
    assert len(recortados) < len(mensajes)


@pytest.mark.parametrize(
    "answer,user_msg,expected",
    [
        ("Respuesta muy corta.", "hola", False),
        ("Respuesta " + "larga " * 200, "hola", True),  # por longitud
        ("Respuesta normal.", "Cuéntame tu experiencia laboral", True),  # por keyword
    ],
)
def test_should_evaluate(answer, user_msg, expected):
    assert should_evaluate(answer, user_msg) == expected        