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

# LangSmith imports
from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log


class ModelLoader:
    """Loads embedding models and LLMs with LangSmith tracing."""
    
    def __init__(self):
        load_dotenv()
        self.config = self._load_config()
        self.api_keys = self._load_api_keys()
        self._setup_langsmith()
        
    def _setup_langsmith(self):
        """Setup LangSmith tracing from .env."""
        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
            api_key = os.getenv("LANGSMITH_API_KEY")
            if api_key:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = api_key
                os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "document-assistant-rag")
                os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
                log.info("LangSmith tracing enabled", 
                        project=os.getenv("LANGSMITH_PROJECT"),
                        endpoint=os.getenv("LANGSMITH_ENDPOINT"))
            else:
                log.warning("LANGSMITH_API_KEY not found in .env. Tracing disabled.")
        
    def _load_config(self) -> dict:
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_api_keys(self) -> dict:
        keys = {}
        
        # Load from .env
        for key in ["GOOGLE_API_KEY", "GROQ_API_KEY", "LANGSMITH_API_KEY"]:
            value = os.getenv(key)
            if value:
                keys[key] = value
                log.debug(f"Loaded {key} from .env")
                
        return keys
    
    def load_embeddings(self) -> Embeddings:
        """Load embedding model."""
        embedding_cfg = self.config.get("embedding_model", {})
        provider = embedding_cfg.get("provider", "huggingface")
        model_name = embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        
        log.info("Loading embedding model", provider=provider, model=model_name)
        
        if provider == "google":
            api_key = self.api_keys.get("GOOGLE_API_KEY")
            if not api_key:
                raise DocumentPortalException(
                    "GOOGLE_API_KEY not found in .env. Please add it."
                )
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=model_name
            )
        else:
            raise DocumentPortalException(f"Unsupported embedding provider: {provider}")
    
    def load_llm(self) -> BaseChatModel:
        """Load LLM with LangSmith tracing."""
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "groq")
        provider_cfg = llm_cfg.get(provider, {})
        model_name = provider_cfg.get("model_name", "llama-3.3-70b-versatile")
        temperature = provider_cfg.get("temperature", 0)
        
        log.info("Loading LLM", provider=provider, model=model_name)
        
        # Get callbacks for tracing
        callbacks = []
        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
            tracer = LangChainTracer(
                project_name=os.getenv("LANGSMITH_PROJECT", "document-assistant-rag")
            )
            callbacks.append(tracer)
            callbacks.append(StreamingStdOutCallbackHandler())
            log.debug("LangSmith callbacks added to LLM")
        
        if provider == "google":
            api_key = self.api_keys.get("GOOGLE_API_KEY")
            if not api_key:
                raise DocumentPortalException(
                    "GOOGLE_API_KEY not found in .env. Please add it."
                )
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                callbacks=callbacks if callbacks else None
            )
        elif provider == "groq":
            api_key = self.api_keys.get("GROQ_API_KEY")
            if not api_key:
                raise DocumentPortalException(
                    "GROQ_API_KEY not found in .env. Please add it."
                )
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                callbacks=callbacks if callbacks else None
            )
        else:
            raise DocumentPortalException(f"Unsupported LLM provider: {provider}")


@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    return ModelLoader()