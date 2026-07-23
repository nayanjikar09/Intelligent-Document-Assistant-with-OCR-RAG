import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log


class ConversationalRAG:
    """LCEL-based Conversational RAG."""
    
    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id
            self.llm = ModelLoader().load_llm()
            
            # Prompts
            self.contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a conversation history and the most recent user query, rewrite the query as a standalone question."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            self.qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant. Answer using the context. If not found, say 'I don't know.'\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()
                
            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize", error=str(e))
            raise DocumentPortalException(e)

    def load_retriever_from_faiss(self, index_path: str, k: int = 5, index_name: str = "faiss_index",
                                   search_type: str = "mmr", fetch_k: int = 20, lambda_mult: float = 0.5,
                                   search_kwargs: Optional[Dict[str, Any]] = None):
        """Load FAISS retriever."""
        try:
            # Convert to Path
            index_path = Path(index_path)
            
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")
            
            # Check multiple possible locations for index files
            # Location 1: index_path / index_name / index_name.faiss (subfolder)
            index_subfolder = index_path / index_name
            index_file_1 = index_subfolder / f"{index_name}.faiss"
            pkl_file_1 = index_subfolder / f"{index_name}.pkl"
            
            # Location 2: index_path / index_name.faiss (direct)
            index_file_2 = index_path / f"{index_name}.faiss"
            pkl_file_2 = index_path / f"{index_name}.pkl"
            
            # Determine which path to use
            if index_file_1.exists() and pkl_file_1.exists():
                load_path = index_subfolder
                log.info("Found FAISS index in subfolder", path=str(load_path))
            elif index_file_2.exists() and pkl_file_2.exists():
                load_path = index_path
                log.info("Found FAISS index directly", path=str(load_path))
            else:
                # List all files in directory for debugging
                all_files = list(index_path.glob("*")) if index_path.exists() else []
                raise FileNotFoundError(
                    f"FAISS index files not found in {index_path}.\n"
                    f"Tried:\n"
                    f"  - {index_file_1}\n"
                    f"  - {index_file_2}\n"
                    f"Files in directory: {[f.name for f in all_files]}"
                )
            
            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                str(load_path), 
                embeddings, 
                index_name=index_name,
                allow_dangerous_deserialization=True
            )
            
            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs["fetch_k"] = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult
            
            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()
            log.info("FAISS retriever loaded", index_path=str(load_path), search_type=search_type)
            return self.retriever
        except Exception as e:
            log.error("Failed to load retriever", error=str(e))
            raise DocumentPortalException(e)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the RAG chain."""
        try:
            if self.chain is None:
                raise DocumentPortalException("RAG chain not initialized. Call load_retriever_from_faiss() first.")
            
            chat_history = chat_history or []
            answer = self.chain.invoke({"input": user_input, "chat_history": chat_history})
            
            if not answer:
                return "No answer generated."
                
            log.info("Chain invoked", session_id=self.session_id, preview=str(answer)[:150])
            return answer
        except Exception as e:
            log.error("Failed to invoke", error=str(e))
            raise DocumentPortalException(e)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        """Build the LCEL chain."""
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set")
            
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            
            retrieve_docs = question_rewriter | self.retriever | self._format_docs
            
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            log.info("LCEL chain built", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build chain", error=str(e))
            raise DocumentPortalException(e)