import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import yaml

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log


class ModelLoader:
    """Loads embedding models and LLMs."""
    
    def __init__(self):
        load_dotenv()
        self.config = self._load_config()
        self.api_keys = self._load_api_keys()
        
    def _load_config(self) -> dict:
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_api_keys(self) -> dict:
        keys = {}
        
        secret = os.getenv("apikeyliveclass")
        if secret:
            try:
                keys.update(json.loads(secret))
            except:
                pass
        
        for key in ["GOOGLE_API_KEY", "GROQ_API_KEY"]:
            value = os.getenv(key)
            if value:
                keys[key] = value
                
        return keys
    
    def load_embeddings(self) -> Embeddings:
        """Load embedding model."""
        embedding_cfg = self.config.get("embedding_model", {})
        provider = embedding_cfg.get("provider", "huggingface")  # Default to huggingface
        model_name = embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        
        log.info("Loading embedding model", provider=provider, model=model_name)
        
        if provider == "google":
            api_key = self.api_keys.get("GOOGLE_API_KEY")
            if not api_key:
                raise DocumentPortalException(
                    "GOOGLE_API_KEY not found. Please set it in .env file."
                )
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
        elif provider == "huggingface":
            # HuggingFace embeddings - no API key required!
            return HuggingFaceEmbeddings(
                model_name=model_name
            )
        else:
            raise DocumentPortalException(f"Unsupported embedding provider: {provider}")
    
    def load_llm(self) -> BaseChatModel:
        """Load LLM."""
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "groq")
        provider_cfg = llm_cfg.get(provider, {})
        model_name = provider_cfg.get("model_name", "llama-3.3-70b-versatile")
        temperature = provider_cfg.get("temperature", 0)
        
        log.info("Loading LLM", provider=provider, model=model_name)
        
        if provider == "google":
            api_key = self.api_keys.get("GOOGLE_API_KEY")
            if not api_key:
                raise DocumentPortalException(
                    "GOOGLE_API_KEY not found. Please set it in .env file."
                )
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature
            )
        elif provider == "groq":
            api_key = self.api_keys.get("GROQ_API_KEY")
            if not api_key:
                raise DocumentPortalException(
                    "GROQ_API_KEY not found. Please set it in .env file."
                )
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=temperature
            )
        else:
            raise DocumentPortalException(f"Unsupported LLM provider: {provider}")


@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    return ModelLoader()