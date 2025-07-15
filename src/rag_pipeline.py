import os
import logging
from typing import Dict, List
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """A complete RAG pipeline with document loading, vector storage, and query capabilities."""
    
    def __init__(self, config: Dict):
        self.config = config
        self._validate_config()
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            length_function=len,
            is_separator_regex=False,
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            show_progress_bar=True
        )
        
        self.llm = ChatOpenAI(
            model_name=config.get("model_name", "gpt-3.5-turbo"),
            temperature=config.get("temperature", 0.7),
            streaming=True,
            max_retries=3
        )
        
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self._initialized = False

    def _validate_config(self):
        """Validate the configuration dictionary."""
        required_keys = ["model_name"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def initialize_pipeline(self, document_path: str) -> bool:
        """
        Initialize the RAG pipeline with a document.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            bool: True if initialization succeeded
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            # 1. Load document
            logger.info(f"Loading document from: {document_path}")
            loader = self._get_document_loader(document_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No documents loaded - file may be empty or corrupted")
            
            # 2. Split documents
            logger.info("Splitting documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            
            # 3. Create vector store
            logger.info("Creating vector store")
            if self.config.get("use_chroma", False):
                persist_dir = self.config.get("persist_directory", "chroma_db")
                os.makedirs(persist_dir, exist_ok=True)
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=persist_dir
                )
            else:
                self.vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
            
            # 4. Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 8}
            )
            
            # 5. Setup QA chain
            self.qa_chain = self._create_qa_chain()
            
            self._initialized = True
            logger.info("RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}", exc_info=True)
            raise Exception(f"RAG initialization failed: {str(e)}")

    def _get_document_loader(self, file_path: str):
        """Get appropriate document loader with error handling."""
        try:
            # Verify file exists and is readable
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at: {file_path}")
            
            if os.path.getsize(file_path) == 0:
                raise ValueError("Uploaded file is empty")

            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.txt':
                # Try multiple encodings for text files
                for encoding in ['utf-8', 'windows-1252', 'iso-8859-1']:
                    try:
                        return TextLoader(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Failed to decode text file with common encodings")
            else:
                return UnstructuredFileLoader(file_path)
                
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise Exception(f"Failed to load document: {str(e)}")

    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}

        Answer in the following format:
        - First provide a concise direct answer
        - Then explain your reasoning in 1-2 sentences
        - Finally cite the relevant source sections
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> Dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dict: {
                "answer": str,
                "source_documents": List[Document],
                "search_results": List[Document]
            }
        """
        if not self._initialized:
            raise Exception("RAG pipeline not initialized. Call initialize_pipeline() first.")
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Get search results
            search_results = self.retriever.invoke(question)
            
            # Get full answer
            answer = self.qa_chain.invoke(question)
            
            return {
                "answer": answer,
                "source_documents": search_results,
                "search_results": search_results
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            raise Exception(f"Query failed: {str(e)}")

    def is_initialized(self) -> bool:
        """Check if pipeline is initialized."""
        return self._initialized

    def clear(self):
        """Clear the pipeline resources."""
        if isinstance(self.vector_store, Chroma):
            self.vector_store.delete_collection()
        self._initialized = False
        logger.info("RAG pipeline cleared")