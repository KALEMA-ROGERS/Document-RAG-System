import tempfile
import shutil
import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix path resolution
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.rag_pipeline import RAGPipeline
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.error("Please ensure:")
    st.error("1. All requirements are installed (pip install -r requirements.txt)")
    st.error("2. You have __init__.py in src/ directory")
    st.stop()

# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Document RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "document_initialized" not in st.session_state:
    st.session_state.document_initialized = False
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None

# Sidebar with document upload and configuration
with st.sidebar:
    st.title("Configuration")

    # Document upload section
    uploaded_file = st.file_uploader(
        "📄 Upload a text document",
        type=["txt"],
        help="Supported formats: .txt files only"
    )

    # File validation
    if uploaded_file and uploaded_file.size > 50_000_000:  # 50MB limit
        st.warning("File too large (max 50MB)")
        uploaded_file = None

    # Model configuration
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "🤖 LLM Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0
    )

    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    # Document processing settings
    st.subheader("Document Processing")
    chunk_size = st.slider(
        "✂️ Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "↔️ Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50
    )

    use_chroma = st.checkbox(
        "💾 Use ChromaDB (persistent storage)",
        value=False
    )

    if st.button("🚀 Initialize/Reinitialize System", type="primary"):
        if uploaded_file is not None:
            # Clean up previous temp directory if exists
            if st.session_state.temp_dir:
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)

            # Create new temp directory
            temp_dir = tempfile.mkdtemp()
            st.session_state.temp_dir = temp_dir
            
            try:
                document_path = os.path.join(temp_dir, uploaded_file.name)

                # Save uploaded file with original name
                with open(document_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Verify file was saved
                if not os.path.exists(document_path):
                    raise FileNotFoundError(f"File failed to save at {document_path}")

                # Initialize RAG pipeline
                config = {
                    "model_name": model_name,
                    "temperature": temperature,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "use_chroma": use_chroma,
                    "persist_directory": "chroma_db" if use_chroma else None
                }

                st.session_state.rag_pipeline = RAGPipeline(config)
                st.session_state.rag_pipeline.initialize_pipeline(document_path)
                st.session_state.document_initialized = True
                st.success("Document processing complete! You can now ask questions.")

            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                st.session_state.temp_dir = None
                st.error(f"Error initializing system: {str(e)}")
                st.session_state.document_initialized = False
        else:
            st.warning("Please upload a document first.")

# Main chat interface
st.title("📚 Document RAG System")
st.caption("Ask questions about your uploaded document")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.document_initialized:
        st.warning("Please upload and initialize a document first.")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from RAG system
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.rag_pipeline.query(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

                # Show source documents if available
                if response.get("source_documents"):
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.caption(f"Source {i+1}:")
                            st.text(doc.page_content[:500] + "...")
                            st.text(f"Metadata: {doc.metadata}")

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]})
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")

# Clear chat button
if st.button("Clear Chat"):
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        st.session_state.temp_dir = None
    st.session_state.messages = []
    st.rerun()