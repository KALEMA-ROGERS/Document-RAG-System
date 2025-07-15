from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class LLMIntegration:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    def create_qa_chain(self, vector_store):
        """Create a retrieval QA chain with the given vector store."""
        try:
            # Custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            chain_type_kwargs = {"prompt": PROMPT}
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True
            )
            
            return qa_chain
        except Exception as e:
            raise Exception(f"Error creating QA chain: {str(e)}")
        
        