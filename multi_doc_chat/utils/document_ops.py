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

# Try multiple ways to import PDF library
PYPDF_AVAILABLE = False
try:
    # Try different import methods
    try:
        import pypdf
        from pypdf import PdfReader
        PYPDF_AVAILABLE = True
        log.info(f"pypdf loaded successfully (version {pypdf.__version__})")
    except ImportError:
        # Try PyPDF2
        try:
            import PyPDF2
            from PyPDF2 import PdfReader
            PYPDF_AVAILABLE = True
            log.info("PyPDF2 loaded successfully")
        except ImportError:
            # Try using importlib
            import importlib
            try:
                pypdf_module = importlib.import_module('pypdf')
                PdfReader = getattr(pypdf_module, 'PdfReader')
                PYPDF_AVAILABLE = True
                log.info("pypdf loaded via importlib")
            except ImportError:
                log.warning("No PDF library found. Install with: pip install pypdf")
except Exception as e:
    log.warning(f"PDF import error: {str(e)}")

# OCR Support
OCR_AVAILABLE = False
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
    log.info("RapidOCR loaded successfully")
except ImportError:
    log.warning("RapidOCR not installed. Image OCR disabled.")
except Exception as e:
    log.warning(f"RapidOCR failed to initialize: {str(e)}. OCR disabled.")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".pptx", ".csv", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff"
}


class OCRDocumentLoader:
    """Extract text from images using OCR."""
    
    def __init__(self):
        if not OCR_AVAILABLE:
            raise DocumentPortalException("RapidOCR not available.")
        try:
            self.ocr = RapidOCR()
        except Exception as e:
            log.error(f"Failed to initialize OCR: {str(e)}")
            raise DocumentPortalException(f"OCR initialization failed: {str(e)}")
    
    def load(self, path: Path) -> List[Document]:
        try:
            result, _ = self.ocr(str(path))
            if not result:
                return []
            text = "\n".join([line[1] for line in result])
            return [Document(
                page_content=text,
                metadata={"source": str(path), "file_type": "image", "ocr_used": True}
            )]
        except Exception as e:
            log.error(f"OCR processing failed: {str(e)}", path=str(path))
            return []


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load documents using appropriate loader based on file extension."""
    
    docs: List[Document] = []
    failed_files: List[str] = []
    
    # Initialize OCR only if available
    ocr_loader = None
    if OCR_AVAILABLE:
        try:
            ocr_loader = OCRDocumentLoader()
        except Exception as e:
            log.warning(f"OCR not available: {str(e)}")

    for p in paths:
        if not p.exists():
            failed_files.append(str(p))
            continue

        ext = p.suffix.lower()
        
        # Handle images with OCR
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            if ocr_loader:
                try:
                    docs.extend(ocr_loader.load(p))
                    log.debug("OCR processed image", path=str(p))
                    continue
                except Exception as e:
                    log.error("OCR failed", path=str(p), error=str(e))
                    failed_files.append(str(p))
                    continue
            else:
                log.warning("OCR not available for image", path=str(p))
                failed_files.append(str(p))
                continue

        # Handle PDF - try multiple methods
        if ext == ".pdf":
            loaded = False
            # Method 1: Try custom loader if available
            if PYPDF_AVAILABLE:
                try:
                    with open(p, 'rb') as file:
                        reader = PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        
                        if text.strip():
                            docs.append(Document(
                                page_content=text,
                                metadata={
                                    "source": str(p),
                                    "file_type": "pdf",
                                    "total_pages": len(reader.pages)
                                }
                            ))
                            log.info(f"PDF loaded with custom loader", path=str(p))
                            loaded = True
                        else:
                            log.warning("No text extracted from PDF", path=str(p))
                except Exception as e:
                    log.warning(f"Custom PDF loader failed: {str(e)}", path=str(p))
            
            # Method 2: Try PyPDFLoader from langchain_community
            if not loaded:
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(p))
                    loaded_docs = loader.load()
                    if loaded_docs:
                        docs.extend(loaded_docs)
                        log.info(f"PDF loaded with PyPDFLoader", path=str(p))
                        loaded = True
                except Exception as e:
                    log.warning(f"PyPDFLoader failed: {str(e)}", path=str(p))
            
            if not loaded:
                failed_files.append(str(p))
            continue

        # Handle other file types
        loader = _get_loader(p, ext)
        if loader:
            try:
                docs.extend(loader.load())
                log.debug("File loaded", path=str(p), ext=ext)
            except Exception as e:
                log.error("Failed to load file", path=str(p), error=str(e))
                failed_files.append(str(p))
        else:
            log.warning("Unsupported file type", path=str(p), ext=ext)
            failed_files.append(str(p))

    if not docs:
        raise DocumentPortalException(f"No readable documents found. Failed: {failed_files}")

    log.info("Documents loaded", count=len(docs), failed=len(failed_files))
    return docs


def _get_loader(path: Path, ext: str) -> Optional[object]:
    """Get appropriate document loader based on file extension."""
    
    loaders = {
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