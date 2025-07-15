from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
import os

class EmbeddingsManager:
    def __init__(self, embedding_type="openai", persist_directory=None):
        self.embedding_type = embedding_type
        self.persist_directory = persist_directory
        
        if embedding_type == "openai":
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def create_vector_store(self, chunks, use_chroma=False):
        """Create a vector store from document chunks."""
        try:
            if use_chroma:
                vector_store = Chroma.from_documents(
                    chunks, 
                    self.embeddings, 
                    persist_directory=self.persist_directory
                )
                vector_store.persist()
            else:
                vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            return vector_store
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
        
        