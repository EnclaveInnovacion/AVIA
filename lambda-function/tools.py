from langchain.agents.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_aws import  AmazonKnowledgeBasesRetriever
import wikipediaapi
import logging

tools_logger = logging.getLogger("TOOLS")
tools_logger.setLevel(logging.INFO)

class Tools:
    """Clase que contiene las herramientas utilizadas por el agente de IA."""
    
    def __init__(self, env_config) -> None:
        self.tool_list = [self.create_kb_retriever_tool(env_config), self.create_wikipedia_tool()]
        
    def create_kb_retriever_tool(self, env_config):
        """Método que crea una herramienta a partir de una base de conocimiento alojada en AWS."""
        kb_retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=env_config.KNOWLEDGE_BASE_ID,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": env_config.RESULTS_NUMBER}},
        )
        
        retriever_tool = create_retriever_tool(
            name="Base de conocimiento de Enclave Formación",
            description="Con esta herramienta podrás acceder a una base de conocimiento sobre Enclave Formación y sus cursos de formación.",
            retriever=kb_retriever
        )
        
        if retriever_tool:
            tools_logger.info("TOOLS: Herramienta de base de conocimiento creada")
            return retriever_tool
        else:
            tools_logger.error("TOOLS: Error al crear la herramienta de base de conocimiento")
    
    def search_wikipedia(self, title):
        """Método que busca en Wikipedia para saber si la página con el título proporcionado existe."""
        wiki_wiki = wikipediaapi.Wikipedia('AVIBot/0.1 (https://t-enclave.com/avi-learning/)','es')
        page = wiki_wiki.page(title)
        
        if page.exists():
            tools_logger.info(f"TOOLS: Página encontrada en Wikipedia:\n{page.summary}")
            return page.summary 
        else:
            tools_logger.info("TOOLS: Página no encontrada en Wikipedia")
            return "No se encontró ninguna página exacta en Wikipedia que coincida con la búsqueda."
        
    def create_wikipedia_tool(self):
        """Método que crea una herramienta de búsqueda en Wikipedia a partir de un método y la biblioteca de 'wikipedia'."""
        wikipedia_tool = Tool.from_function(
            func=self.search_wikipedia,
            name="Wikipedia",
            description="Con esta herramienta podrás acceder a Wikipedia para resumir o explicar temas variados que no conozcas por defecto y que no tengan relación con Enclave Formación."
        )
        
        if wikipedia_tool:
            tools_logger.info("TOOLS: Herramienta de Wikipedia creada")
            return wikipedia_tool
        else:
            tools_logger.error("TOOLS: Error al crear la herramienta de Wikipedia")
    
    