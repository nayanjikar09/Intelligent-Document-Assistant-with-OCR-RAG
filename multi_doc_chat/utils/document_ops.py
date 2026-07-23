from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from fastapi import UploadFile

# OCR Support
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    log.warning("RapidOCR not installed. Image OCR disabled.")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".pptx", ".csv", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff"
}


class OCRDocumentLoader:
    """Extract text from images using OCR."""
    
    def __init__(self):
        if not OCR_AVAILABLE:
            raise DocumentPortalException("RapidOCR not installed.")
        self.ocr = RapidOCR()
    
    def load(self, path: Path) -> List[Document]:
        result, _ = self.ocr(str(path))
        if not result:
            return []
        text = "\n".join([line[1] for line in result])
        return [Document(
            page_content=text,
            metadata={"source": str(path), "file_type": "image", "ocr_used": True}
        )]


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load documents using appropriate loader based on file extension."""
    
    docs: List[Document] = []
    failed_files: List[str] = []
    ocr_loader = OCRDocumentLoader() if OCR_AVAILABLE else None

    for p in paths:
        if not p.exists():
            failed_files.append(str(p))
            continue

        ext = p.suffix.lower()
        
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            if ocr_loader:
                try:
                    docs.extend(ocr_loader.load(p))
                    continue
                except Exception as e:
                    log.error("OCR failed", path=str(p), error=str(e))
                    failed_files.append(str(p))
                    continue
            else:
                failed_files.append(str(p))
                continue

        loader = _get_loader(p, ext)
        if loader:
            try:
                docs.extend(loader.load())
            except Exception as e:
                log.error("Failed to load file", path=str(p), error=str(e))
                failed_files.append(str(p))
        else:
            failed_files.append(str(p))

    if not docs:
        raise DocumentPortalException(f"No readable documents found. Failed: {failed_files}")

    log.info("Documents loaded", count=len(docs), failed=len(failed_files))
    return docs


def _get_loader(path: Path, ext: str) -> Optional[object]:
    """Get appropriate document loader based on file extension."""
    
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
        ".md": UnstructuredMarkdownLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
    }
    
    loader_class = loaders.get(ext)
    return loader_class(str(path)) if loader_class else None


class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to work with file_io utilities."""
    
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.filename = uf.filename or "file"
        self.name = self.filename

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()
    
    def read(self) -> bytes:
        return self.getbuffer()
    
    def close(self) -> None:
        self._uf.file.close()