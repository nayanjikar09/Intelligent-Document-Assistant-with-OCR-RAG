# 📄 MultiDocChat - Intelligent Document Assistant with OCR & RAG

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![UV](https://img.shields.io/badge/UV-Package%20Manager-purple.svg)](https://docs.astral.sh/uv/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-red.svg)](https://docs.pydantic.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow.svg)](https://github.com/facebookresearch/faiss)

## 📌 Overview

MultiDocChat is an **Intelligent Document Assistant** that combines **OCR (Optical Character Recognition)** with **RAG (Retrieval-Augmented Generation)** to provide intelligent answers from your documents. Upload any document and start asking questions in natural language!

### 🎯 Key Features

- ✅ **Multi-format Support** - PDF, DOCX, TXT, Images (PNG, JPG, BMP, TIFF)
- ✅ **OCR Integration** - Extracts text from images and scanned documents using RapidOCR
- ✅ **RAG Pipeline** - Uses FAISS vector search with MMR for accurate retrieval
- ✅ **Chat Interface** - Interactive chat with document context and history
- ✅ **Session Management** - Maintains conversation history per session
- ✅ **Dark Theme UI** - Modern, responsive black & blue interface
- ✅ **Multiple LLM Support** - Groq, Google Gemini, or HuggingFace
- ✅ **LangSmith Tracing** - Monitor and debug your RAG pipeline
- ✅ **Pydantic Validation** - Type-safe data validation for all models
- ✅ **MMR Search** - Maximal Marginal Relevance for diverse results

## 🎬 Demo Video

> **Watch the full demo below:**

https://github.com/user-attachments/assets/30b9d62d-b280-4459-9a75-5ad5fb7c2d18

*Click the video above to watch the demonstration*

## 🏗️ Architecture



## 🛠️ Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | FastAPI | High-performance API with async support |
| **Data Validation** | Pydantic V2 | Type-safe request/response models |
| **RAG Framework** | LangChain | Orchestrating the RAG pipeline |
| **LLM** | Groq (llama-3.3-70b) | High-speed inference |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Text vectorization |
| **Vector Store** | FAISS | Efficient similarity search |
| **Retrieval Strategy** | MMR (Maximal Marginal Relevance) | Diverse and relevant results |
| **OCR** | RapidOCR | Text extraction from images |
| **Logging** | Structlog | Structured JSON logging |
| **Tracing** | LangSmith | Pipeline monitoring and debugging |
| **Package Manager** | UV | Fast dependency management |

### Frontend
| Component | Technology |
|-----------|------------|
| **UI Framework** | Vanilla HTML5/CSS3/JavaScript |
| **Theme** | Dark theme with blue accents |
| **Icons** | Font Awesome 6 |
| **Styling** | CSS3 with animations |
| **API Client** | Fetch API |

### Document Processing
| Feature | Technology |
|---------|------------|
| **PDF** | PyPDF / PyPDF2 / PDFPlumber |
| **Word** | Docx2txt |
| **Text** | TextLoader |
| **PowerPoint** | UnstructuredPowerPointLoader |
| **Excel** | UnstructuredExcelLoader |
| **Images** | RapidOCR with ONNX Runtime |
| **Markdown** | UnstructuredMarkdownLoader |
| **CSV** | CSVLoader |

## 📁 Project Structure
Intelligent-Document-Assistant-with-OCR-RAG/
├── multi_doc_chat/
│ ├── init.py
│ ├── exception/
│ │ ├── init.py
│ │ └── custom_exception.py # Custom exception handling
│ ├── logger/
│ │ ├── init.py
│ │ └── custom_logger.py # Structured logging with structlog
│ ├── utils/
│ │ ├── init.py
│ │ ├── document_ops.py # Document loading & OCR
│ │ ├── file_io.py # File upload/save utilities
│ │ └── model_loader.py # LLM & embedding model loading
│ └── src/
│ ├── init.py
│ ├── document_ingestion/
│ │ ├── init.py
│ │ └── data_ingestion.py # ChatIngestor with FAISS indexing
│ └── document_chat/
│ ├── init.py
│ └── retrieval.py # ConversationalRAG with LCEL
├── templates/
│ └── index.html # Main UI template
├── static/
│ ├── style.css # Dark theme styling
│ └── script.js # Frontend logic
├── config/
│ └── config.yaml # Configuration file
├── data/ # Uploaded documents storage
├── faiss_index/ # FAISS vector indices
├── logs/ # Application logs
├── main.py # FastAPI entry point
├── test.py # Test script
├── requirements.txt # Pip dependencies
├── pyproject.toml # Project metadata
├── uv.lock # Locked dependencies
├── .env # Environment variables
└── README.md # Documentation