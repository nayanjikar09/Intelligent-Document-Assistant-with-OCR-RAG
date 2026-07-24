import sys
print(f"Python: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import pypdf
    print(f"✅ pypdf version: {pypdf.__version__}")
    from pypdf import PdfReader
    print("✅ PdfReader imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")