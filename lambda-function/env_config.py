import os
import boto3

REGION_NAME = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("MODEL_ID","anthropic.claude-v2:1")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5")) # Aleatoriedad
TOP_K = int(os.getenv("TOP_K", "50")) # Limita predicciones
TOP_P = float(os.getenv("TOP_P", "0.9")) # Probabilidad de tokens generados
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000")) # Limita el num de tokens

RESULTS_NUMBER = int(os.getenv("RESULTS_NUMBER", "4")) # Num. resultados de base de conocimiento

BOTO3_SESSION = boto3.Session(region_name=REGION_NAME)
BEDROCK_CLIENT = BOTO3_SESSION.client("bedrock-runtime", region_name=REGION_NAME)
