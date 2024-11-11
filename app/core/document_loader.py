from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import boto3
from docx import Document
from io import BytesIO
import fitz

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

class S3DocumentLoader:
    def __init__(self):
        self.client = boto3.client('s3')
        self.chunk_size = 512
        self.chunk_overlap = 50

    def load_documents(self, bucket_name, file_key):
        # Retrieve the file from S3
        obj = self.client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = obj["Body"].read()
        
        # If it's a PDF, use PyMuPDF to extract text
        if file_key.endswith(".pdf"):
            pdf = fitz.open(stream=file_content, filetype="pdf")
            corpus = "\n".join([page.get_text() for page in pdf])
        # If it's a Word document, use python-docx to extract text
        elif file_key.endswith(".docx"):
            doc = Document(BytesIO(file_content))
            corpus = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError("Unsupported file format")

        # Prepare the text for splitting
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Wrap corpus in a DocumentText object with both page_content and metadata attributes
        class DocumentText:
            def __init__(self, text):
                self.page_content = text
                self.metadata = {}  # Add an empty metadata dictionary

        # Create a DocumentText instance with the extracted corpus
        doc_object = DocumentText(corpus)
        
        # Split the document into chunks
        return text_splitter.split_documents([doc_object])

# Example usage
''' bucket_name = 'training-data-hopfalgebra-slm'
object_key = 'training-data/example_training_data.pdf'
s3_client = S3DocumentLoader()
corpus = s3_client.load_documents(bucket_name, object_key)
print(corpus) '''
