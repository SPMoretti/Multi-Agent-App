#llm.py

from crewai import LLM
from config import MODEL_ANALYST, MODEL_FILE_MANAGER, MODEL_RESEARCHER, MODEL_ROUTER

def get_researcher_llm():
    return LLM(model=MODEL_RESEARCHER, temperature=0.0, max_tokens=1024)

def get_file_manager_llm():
    return LLM(model=MODEL_FILE_MANAGER, temperature=0.0, max_tokens=512)

def get_analyst_llm():
    return LLM(model=MODEL_ANALYST, temperature=0.2, max_tokens=2048)

def get_router_llm():
    return LLM(model=MODEL_ROUTER, temperature=0.0, max_tokens=64)
