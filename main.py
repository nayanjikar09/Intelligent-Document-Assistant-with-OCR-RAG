from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log


# ----------------------------
# FastAPI initialization
# ----------------------------
app = FastAPI(
    title="MultiDocChat - OCR & RAG",
    description="Intelligent Document Assistant with OCR and RAG",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

# Create directories if they don't exist
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ----------------------------
# In-memory storage
# ----------------------------
SESSIONS: Dict[str, List[dict]] = {}


# ----------------------------
# Adapters
# ----------------------------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to work with ChatIngestor."""
    
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()
    
    @property
    def filename(self):
        return self.name


# ----------------------------
# Pydantic Models
# ----------------------------
class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str
    file_count: int = 0
    ocr_used: bool = False


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    history_length: int = 0


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "MultiDocChat API with OCR is running",
        "version": "1.0.0"
    }


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    """Render the home page."""
    try:
        html_path = templates_dir / "index.html"
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            return HTMLResponse(content="<h1>Template not found</h1>", status_code=404)
    except Exception as e:
        log.error("Template error", error=str(e))
        return HTMLResponse(content=f"<h1>Error loading template</h1><p>{str(e)}</p>", status_code=500)


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload documents and create FAISS index.
    
    Supports:
    - PDF, DOCX, TXT (direct text extraction)
    - Images (PNG, JPG, BMP, TIFF) → OCR text extraction
    - PPTX, CSV, XLSX, MD
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # Detect if any image files are uploaded
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        ocr_used = any(
            Path(f.filename or "").suffix.lower() in image_extensions 
            for f in files
        )
        
        if ocr_used:
            log.info("📸 Image files detected - OCR will be used")

        # Wrap files
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        # Initialize ingestor
        ingestor = ChatIngestor(
            temp_base="data",
            faiss_base="faiss_index",
            use_session_dirs=True
        )
        session_id = ingestor.session_id

        # Build retriever with MMR
        retriever = ingestor.build_retriever(
            uploaded_files=wrapped_files,
            chunk_size=1000,
            chunk_overlap=200,
            k=5,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Initialize session history
        SESSIONS[session_id] = []

        return UploadResponse(
            session_id=session_id,
            indexed=True,
            message=f"Successfully indexed {len(files)} file(s) {'with OCR' if ocr_used else ''}",
            file_count=len(files),
            ocr_used=ocr_used
        )

    except DocumentPortalException as e:
        log.error("Upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error("Upload error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the RAG system with session context."""
    session_id = req.session_id
    message = req.message.strip()

    if not session_id or session_id not in SESSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired session. Please upload documents first."
        )

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Load RAG with FAISS index
        rag = ConversationalRAG(session_id=session_id)
        index_path = f"faiss_index/{session_id}"
        
        rag.load_retriever_from_faiss(
            index_path=index_path,
            k=5,
            index_name="faiss_index",
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Get chat history
        simple_history = SESSIONS.get(session_id, [])
        lc_history = []
        for m in simple_history:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        # Get answer
        answer = rag.invoke(message, chat_history=lc_history)

        # Update history
        simple_history.append({"role": "user", "content": message})
        simple_history.append({"role": "assistant", "content": answer})
        SESSIONS[session_id] = simple_history

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            history_length=len(simple_history)
        )

    except DocumentPortalException as e:
        log.error("Chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error("Chat error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/history/{session_id}")
async def get_history(session_id: str) -> JSONResponse:
    """Get chat history for a session."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return JSONResponse({
        "session_id": session_id,
        "history": SESSIONS.get(session_id, []),
        "total_messages": len(SESSIONS.get(session_id, []))
    })


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    """Delete a session and its history."""
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        # Also delete FAISS index
        index_path = Path(f"faiss_index/{session_id}")
        if index_path.exists():
            shutil.rmtree(index_path)
        return JSONResponse({"message": f"Session {session_id} deleted successfully"})
    
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions() -> JSONResponse:
    """List all active sessions."""
    sessions = []
    for sid, history in SESSIONS.items():
        sessions.append({
            "session_id": sid,
            "message_count": len(history),
            "created_at": sid.split("_")[1] if "_" in sid else "unknown"
        })
    return JSONResponse({"sessions": sessions, "total": len(sessions)})


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )