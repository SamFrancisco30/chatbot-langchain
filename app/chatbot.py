from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def create_retrieval_qa_chain(llm, vector_store):
    """
    Create a RetrievalQA chain for querying the vector store.
    """
#     prompt = ChatPromptTemplate.from_template("""Document: \n{context}\n  Question: \n{input}\n INSTRUCTIONS:
# You are a helpful assistant for the [Software Name] application. Your task is to answer the user's question based on the provided DOCUMENT text. Follow these guidelines: 
#     1. Accuracy: Ensure your answer is grounded in the facts of the DOCUMENT. Do not speculate or provide information outside the document. 2. Relevance: Focus on answering the user's specific question about the software. Avoid unnecessary details.
#     3. Clarity: Provide a clear and concise answer that directly addresses the user's question. Avoid unnecessary jargon or technical terms. 4. Completeness: Provide a complete answer to the user's question. If the question has multiple parts, ensure you address each part.
#                                               5. Tone: Maintain a friendly and professional tone throughout the conversation. Avoid being overly formal or casual. 6. Handling Ambiguity: If the question is unclear or the document does not provide enough information, ask the user for clarification or additional details.""")
    
    prompt = ChatPromptTemplate.from_template("""Answer the question concisely using the provided context.\n\n Context: \n{context}\n  Question: \n{input}\n\n  Answer: """)
    combined_documents_chain = create_stuff_documents_chain(
        llm, prompt
    )
    retrieval_chain  = create_retrieval_chain(
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain =combined_documents_chain,
    )
    return retrieval_chain

def load_llm(model_path):
    """
    Load the Llama-2-7B-Chat GGUF model using llama-cpp-python.
    """
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=1,
        n_batch=512,
        n_ctx=4096,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
    )
    return llm