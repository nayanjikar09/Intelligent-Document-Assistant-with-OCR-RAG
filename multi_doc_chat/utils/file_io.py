from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Iterable, List

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls",
    ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff",
}


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """Save uploaded files to disk."""
    
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    for uploaded_file in uploaded_files:
        original_name = getattr(uploaded_file, "filename", getattr(uploaded_file, "name", "uploaded_file"))
        extension = Path(original_name).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            log.warning("Unsupported file skipped", filename=original_name, extension=extension)
            continue

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(original_name).stem).lower()
        unique_filename = f"{safe_name}_{uuid.uuid4().hex[:8]}{extension}"
        output_path = target_dir / unique_filename

        if hasattr(uploaded_file, "file"):
            uploaded_file.file.seek(0)
            data = uploaded_file.file.read()
        elif hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
        elif callable(getattr(uploaded_file, "getbuffer", None)):
            data = uploaded_file.getbuffer()
        else:
            raise ValueError("Unsupported uploaded file object.")

        if isinstance(data, memoryview):
            data = data.tobytes()

        with output_path.open("wb") as file:
            file.write(data)
        
        saved_files.append(output_path)
        log.info("File saved", original=original_name, saved=str(output_path))

    log.info("Upload completed", total=len(saved_files))
    return saved_files