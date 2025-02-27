import os
import time
from document_loader import load_and_split_pdfs
from vector_store import create_vector_store, save_vector_store
from chatbot import create_retrieval_qa_chain, load_llm
from langchain_redis import RedisCache
from langchain.globals import set_llm_cache

def timed_completion(chain, user_input):
    start_time = time.time()
    result = chain.invoke({"input": user_input})
    end_time = time.time()
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
    redis_cache = RedisCache(redis_url=REDIS_URL)
    set_llm_cache(redis_cache)

    retrieve_qa_chain = create_retrieval_qa_chain(llm, vector_store)
    print("Chatbot is ready! Type 'exit' to end the conversation.")
        
    while True:
        user_input = input("\nType in your question: ")
        if user_input.lower() == "exit":
            break

        # response = retrieve_qa_chain.invoke({"input": user_input})
        result, time = timed_completion(retrieve_qa_chain, user_input)
        print(f"Result: {result['answer']}\nTime: {time:.2f} seconds\n")


if __name__ == "__main__":
    main()
