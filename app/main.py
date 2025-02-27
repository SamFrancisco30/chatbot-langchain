import os
import time
from document_loader import load_and_split_pdfs
from vector_store import create_vector_store, save_vector_store
from chatbot import create_retrieval_qa_chain, load_llm
from langchain_redis import RedisSemanticCache
from langchain.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings


def timed_completion(chain, user_input, chat_history):
    start_time = time.time()
    result = chain.invoke({"input": user_input, "chat_history": chat_history})
    end_time = time.time()
    chat_history.extend([HumanMessage(content=user_input), AIMessage(content=result["answer"]),])
    return result, end_time - start_time

def main():
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"Connecting to Redis at: {REDIS_URL}")
    # Load and process PDF documents
    documents = load_and_split_pdfs("data/documents")
    
    # Create and save the vector store
    vector_store = create_vector_store(documents, "data/vector_store")
    save_vector_store(vector_store, "data/vector_store")
    
    # Load the LLM (GGUF format)
    llm = load_llm("models/llama-2-7b-chat.Q4_K_M.gguf")

    # Create a Redis cache instance
    redis_cache = RedisSemanticCache(redis_url=REDIS_URL, 
                                     distance_threshold=0.1,
                                     embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    set_llm_cache(redis_cache)

    rag_chain = create_retrieval_qa_chain(llm, vector_store)
    print("Chatbot is ready! Type 'exit' to end the conversation.")

    chat_history = []
        
    while True:
        user_input = input("\nType in your question: ")
        if user_input.lower() == "exit":
            break

        # response = retrieve_qa_chain.invoke({"input": user_input})
        result, time = timed_completion(rag_chain, user_input, chat_history)
        print(f"\n\nResult:{result['answer']}\n\n Time: {time:.2f} seconds\n")


if __name__ == "__main__":
    main()
