import json
import logging

import env_config
from agent import ReactAgent
from langchain_aws import ChatBedrock
from messages import MessagesDecoder, MessagesEncoder
from tools import Tools


app_logger = logging.getLogger("APP")
app_logger.setLevel(logging.INFO)

llm = ChatBedrock(
    client=env_config.BEDROCK_CLIENT,
    model_id=env_config.MODEL_ID,
    model_kwargs={
            "temperature": env_config.TEMPERATURE,
            "top_k": env_config.TOP_K,
            "top_p": env_config.TOP_P,
            "max_tokens": env_config.MAX_TOKENS,
        },
)

tools = Tools(env_config).tool_list

TEMPLATE = """Eres un asistente de inteligencia artificial llamado AVI que responde preguntas del usuario en el idioma en el que está escrita la pregunta. Las herramientas que puede que necesites para ayudarte son:

[{tools}]

No hace falta que las herramientas sean utilizadas para contestar a la pregunta del usuario, debes utilizarlas cuando no sepas responder por tí mismo. Utilizarás el historial de chat entre etiquetas XML de <chat_history> para ayudarte a contextualizar las preguntas realizadas por el usuario. Para responder de la manera más concisa y acertada tienes que emplear este formato:

<chat_history>
[HumanMessage(content="hola, me llamo pepe")], [AIMessage(content="Hola Pepe, soy AVI")]
</chat_history>

Question: la pregunta realizada por el usuario que debes responder, si no sabes la respuesta di "Lo siento, no conozco la respuesta a tu pregunta"
Thought: este es mi bloc de notas, en el que pongo los pasos que tengo que seguir para responder a la pregunta, además me preguntaré: ¿Tengo que utilizar una herramienta? Sí
Action: la acción que debes realizar, tiene que ser una de: [{tool_names}]
Action Input: el input que debes proporcionar a la herramienta, NUNCA devuelvas el resultado de la herramienta en este paso
Observation: el resultado de la acción, si no devuelve nada di "No lo se"
... (este formato de Thought/Action/Action Input/Observation NO SE PUEDE REPETIR)

Si no necesitas utilizar una herramienta para responder a la pregunta o has conseguido el resultado de realizar la acción DEBES utilizar este formato:

Thought: ¿Tengo que utilizar una herramienta? No
Action: recuperar la respuesta
Final Answer: la respuesta final a la pregunta original del usuario es: [respuesta final]


¡Comienza!

{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


def lex_format_response(event, response_text, chat_history, content_type):
    """
    Crea el formato de respuesta requerido por Lex, incluyendo el historial
    de chat y el tipo de respuesta: SSML (Audio) | PlainText (Texto)
    """
    event["sessionState"]["intent"]["state"] = "Fulfilled"

    if content_type == "SSML":
        app_logger.info("APP: SSML response")
        return {
            "sessionState": {
                "sessionAttributes": {"chat_history": chat_history},
                "dialogAction": {"type": "Close"},
                "intent": event["sessionState"]["intent"],
            },
            "messages": [
                {
                    "contentType": "SSML",
                    "content": f"""
                        <speak>
                            <lang xml:lang="es-ES">
                                {response_text}
                            </lang>
                        </speak>
                    """,
                }
            ],
            "sessionId": event["sessionId"],
            "requestAttributes": (
                event["requestAttributes"] if "requestAttributes" in event else None
            ),
        }

    else:
        app_logger.info("APP: PlainText response")
        return {
            "sessionState": {
                "sessionAttributes": {"chat_history": chat_history},
                "dialogAction": {"type": "Close"},
                "intent": event["sessionState"]["intent"],
            },
            "messages": [
                {
                    "contentType": "PlainText",
                    "content": response_text,
                },
            ],
            "sessionId": event["sessionId"],
            "requestAttributes": (
                event["requestAttributes"] if "requestAttributes" in event else None
            ),
        }


def load_chat_history(session):
    """
    Transforma el json recuperado de la sesión en mensajes de tipo 
    AIMessage o HumanMessage
    """
    if "chat_history" in session:
        return json.loads(session["chat_history"], cls=MessagesDecoder)
    else:
        return []
    
def save_chat_history(chat_history):
    """
    Transforma la lista de mensajes de tipo AIMessage o HumanMessage 
    a un json. Además si hay más 6 mensajes en el historial de chat
    borra los 3 primeros.
    """
    if len(chat_history) > 6:
        chat_history = chat_history[-3:]
    return json.dumps(chat_history, cls=MessagesEncoder)

def lambda_handler(event, context):
    """Función que ejecuta Lambda"""

    if event["inputTranscript"]:
        user_input = event["inputTranscript"]
        session = event["sessionState"]["sessionAttributes"]
        input_mode = event["inputMode"]
        session_id = event["sessionId"]

        content_type = "SSML" if input_mode == "Speech" else "PlainText"
        
        chat_history = load_chat_history(session)
        
        agent = ReactAgent(llm, TEMPLATE, tools, session_id, chat_history)

        if user_input.strip() == "":
            result = {"answer": "Por favor, realiza una pregunta."}
        else:
            input_variables = {"input": user_input, "chat_history": chat_history}

            result = agent.invoke(input_variables)

            answer, chat_history = result
            
        chat_history = save_chat_history(chat_history)

        return lex_format_response(
            event,
            answer,
            chat_history,
            content_type,
        )
