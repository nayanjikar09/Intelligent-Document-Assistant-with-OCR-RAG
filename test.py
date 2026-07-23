import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG

load_dotenv()


def test_document_ingestion_and_rag():
    try:
        # Get base directory
        base_dir = Path(__file__).parent
        
        # Define test files
        test_files = [
            str(base_dir / "data" / "sample.png"),
            str(base_dir / "data" / "Agentic_AI_RAG_Training.txt"),
        ]

        # Open and validate files
        uploaded_files = []
        for file_path in test_files:
            path = Path(file_path)
            if path.exists():
                uploaded_files.append(open(path, "rb"))
                print(f"✓ Added: {file_path}")
            else:
                print(f"✗ File not found: {file_path}")

        if not uploaded_files:
            print("\n❌ No valid files to upload. Exiting.")
            sys.exit(1)

        # Build FAISS index
        print("\n📄 Building retriever...")
        ci = ChatIngestor(
            temp_base="data", 
            faiss_base="faiss_index", 
            use_session_dirs=True,
            index_name="faiss_index"  # Consistent name
        )
        
        retriever = ci.build_retriever(
            uploaded_files, 
            chunk_size=200, 
            chunk_overlap=20, 
            k=5,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Close file handles
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        # Load RAG with FAISS index
        session_id = ci.session_id
        index_dir = os.path.join("faiss_index", session_id)

        print(f"\n🔍 Loading RAG with session: {session_id}")
        print(f"📂 Index directory: {index_dir}")
        
        # Check if index exists
        if not os.path.exists(index_dir):
            print(f"❌ Index directory not found: {index_dir}")
            sys.exit(1)
            
        # List files in index directory
        print(f"📁 Files in index directory: {os.listdir(index_dir)}")
        
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir, 
            k=5, 
            index_name="faiss_index",  # Consistent name
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Start interactive chat
        chat_history = []
        print("\n" + "="*60)
        print("💬 Chat started! Type 'exit' to quit.")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting chat.")
                break

            if not user_input:
                continue
                
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("👋 Goodbye!")
                break

            answer = rag.invoke(user_input, chat_history=chat_history)
            print("Assistant:", answer)
            print()

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_document_ingestion_and_rag()