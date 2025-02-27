from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_pdfs(directory_path):
    """
    Load PDF documents from the specified directory and split them into chunks.
    """
    documents = []
    for file in os.listdir(directory_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(directory_path, file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_documents = text_splitter.split_documents(documents)
    return split_documents