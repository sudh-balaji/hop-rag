from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

class DocumentLoader:
  def __init__(self, file_path: str = "./docs/test.txt"):
    self.file_path = file_path
    self.chunk_size = 512
    self.chunk_overlap = 50
    
  def load_documents(self):
    loader = TextLoader(self.file_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
      chunk_size=self.chunk_size,
      chunk_overlap=self.chunk_overlap
    )
    return text_splitter.split_documents(docs)