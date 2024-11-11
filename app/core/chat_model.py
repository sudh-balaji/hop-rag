from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from .prompt_templates import chat_prompt_template
from .document_loader import DocumentLoader, S3DocumentLoader
from .embeddings import Embeddings
from .model_loader import ModelLoader

class ChatModel:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.document_loader = S3DocumentLoader()
        self.model_loader = ModelLoader()
        self.embeddings = Embeddings(self.model_loader, model_name=model_name)

    def generate_response(self, query: str) -> dict:
        try:
            documents = self.document_loader.load_documents(bucket_name='training-data-hopfalgebra-slm', file_key='training-data/example_training_data.pdf')
            retriever = self.embeddings.load_embeddings(documents)
            
            chain = (
              {"context": retriever,
              "question": RunnablePassthrough()}
              | chat_prompt_template
              | self.embeddings.model
              | StrOutputParser()
            )
            
            response = chain.invoke(query)            
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"