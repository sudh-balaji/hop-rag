from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

class Embeddings:
  def __init__(self,
               model_loader, 
               model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
               embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    self.model_loader = model_loader
    self.embeddings = self.model_loader.load_embedding_model(embedding_model_name)
    self.model = self.model_loader.load_model(model_name)
        
  def load_embeddings(self, documents):
    try:
      db = Chroma.from_documents(documents, self.embeddings)
      k = min(3, len(documents))
      return db.as_retriever(search_kwargs={"k": k})
    except Exception as e:
      raise Exception(f"Error creating retriever: {str(e)}")
