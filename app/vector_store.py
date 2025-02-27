from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_store(documents, path):
    """
    Create a FAISS vector store from the provided documents or load an existing one if it exists.
    """
    if os.path.exists(path):
        return load_vector_store(path)
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)
        save_vector_store(vector_store, path)
        return vector_store

def save_vector_store(vector_store, path):
    """
    Save the FAISS vector store to disk.
    """
    vector_store.save_local(path)

def load_vector_store(path):
    """
    Load a FAISS vector store from disk.
    """
    return FAISS.load_local(path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)