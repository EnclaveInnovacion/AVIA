import json

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

class MessagesEncoder(json.JSONEncoder):
    """Codificador de mensajes a json"""

    def default(self, o):
        """Codifica un mensaje a json a partir de un objeto"""
        if isinstance(o, HumanMessage):
            return {"type": "human", "content": o.content}
        elif isinstance(o, AIMessage):
            return {"type": "ai", "content": o.content}
        else:
            return super().default(o)


class MessagesDecoder(json.JSONDecoder):
    """Decodificador de mensajes desde json"""

    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.decode_message)

    def decode_message(self, obj):
        """Decodifica un mensaje a partir de un objeto json"""
        if "type" in obj and "content" in obj:
            if obj["type"] == "human":
                return HumanMessage(content=obj["content"])
            elif obj["type"] == "ai":
                return AIMessage(content=obj["content"])
        return obj
