# app/config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGENT_MODEL = os.getenv("AGENT_MODEL")
EVAL_MODEL = os.getenv("EVAL_MODEL")
PDF_PATH = os.getenv("PDF_PATH")
SUMMARY_PATH = os.getenv("SUMMARY_PATH")
NOMBRE = os.getenv("NOMBRE")
