from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.document_ops import load_documents
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.model_loader import ModelLoader


def generate_session_id() -> str:
    """Generate unique session ID."""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class FaissManager:
    """Manages FAISS vector store operations."""

    def __init__(self, index_dir: Path, model_loader: ModelLoader, index_name: str = "faiss_index"):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = model_loader.load_embeddings()
        self.index_name = index_name
        self.index_path = self.index_dir / index_name
        self.vector_store: Optional[FAISS] = None

    def load_or_create(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> FAISS:
        """Load existing FAISS or create new one."""
        
        # Check if the index folder exists with files inside
        if self.index_path.exists() and (self.index_path / f"{self.index_name}.faiss").exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.index_path), 
                    self.embeddings, 
                    index_name=self.index_name,
                    allow_dangerous_deserialization=True
                )
                log.info("Loaded FAISS index", path=str(self.index_path))
                return self.vector_store
            except Exception as e:
                log.warning("Failed to load FAISS index, creating new one", error=str(e))
                import shutil
                shutil.rmtree(self.index_dir)
                self.index_dir.mkdir(parents=True, exist_ok=True)

        # Create new index
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        self.vector_store.save_local(str(self.index_path), index_name=self.index_name)
        log.info("Created FAISS index", documents=len(texts), path=str(self.index_path))
        return self.vector_store

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to existing FAISS index."""
        if not self.vector_store:
            raise DocumentPortalException("FAISS not initialized. Call load_or_create() first.")
        if not documents:
            return 0
            
        self.vector_store.add_documents(documents)
        self.vector_store.save_local(str(self.index_path), index_name=self.index_name)
        log.info("Added documents to FAISS", count=len(documents))
        return len(documents)


class ChatIngestor:
    """Handles document ingestion, chunking, and FAISS indexing."""

    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
        index_name: str = "faiss_index",
    ):
        self.model_loader = ModelLoader()
        self.config = self.model_loader.config
        self.session_id = session_id or generate_session_id()
        self.use_session = use_session_dirs
        self.index_name = index_name
        
        self.temp_base = Path(temp_base)
        self.temp_base.mkdir(parents=True, exist_ok=True)
        
        self.faiss_base = Path(faiss_base)
        self.faiss_base.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = self._resolve_dir(self.temp_base)
        self.faiss_dir = self._resolve_dir(self.faiss_base)
        
        log.info("ChatIngestor initialized", session=self.session_id)

    def _resolve_dir(self, base_dir: Path) -> Path:
        if self.use_session:
            session_dir = base_dir / self.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            return session_dir
        return base_dir

    def _split(self, documents: List[Document], chunk_size: Optional[int] = None, 
               chunk_overlap: Optional[int] = None) -> List[Document]:
        chunk_size = chunk_size or self.config.get("chunking", {}).get("chunk_size", 1000)
        chunk_overlap = chunk_overlap or self.config.get("chunking", {}).get("chunk_overlap", 200)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        
        log.info("Documents split", documents=len(documents), chunks=len(chunks))
        return chunks

    def build_retriever(self, uploaded_files: Iterable, **kwargs):
        try:
            retriever_cfg = self.config.get("retriever", {})
            k = kwargs.get("k") or retriever_cfg.get("top_k", 5)
            search_type = kwargs.get("search_type") or retriever_cfg.get("search_type", "mmr")
            fetch_k = kwargs.get("fetch_k") or retriever_cfg.get("fetch_k", 20)
            lambda_mult = kwargs.get("lambda_mult") or retriever_cfg.get("lambda_mult", 0.5)

            saved_paths = save_uploaded_files(uploaded_files, self.temp_dir)
            if not saved_paths:
                raise DocumentPortalException("No supported files uploaded.")

            documents = load_documents(saved_paths)
            if not documents:
                raise DocumentPortalException("No readable documents found.")
            
            chunks = self._split(documents, kwargs.get("chunk_size"), kwargs.get("chunk_overlap"))

            faiss_manager = FaissManager(self.faiss_dir, self.model_loader, self.index_name)
            vector_store = faiss_manager.load_or_create(
                texts=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks]
            )
            faiss_manager.add_documents(chunks)

            search_kwargs = {"k": k}
            if search_type == "mmr":
                search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})

            retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            
            log.info("Retriever created", search_type=search_type)
            return retriever

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException(e) from e