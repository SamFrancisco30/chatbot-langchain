from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

def create_retrieval_qa_chain(llm, vector_store):
    """
    Create a RetrievalQA chain for querying the vector store.
    """
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    system_prompt = (
        "Based on the following context, answer the question concisely but with enough details."
        "\n\n"
        "Context:\n{context}"
        "\n\n"
        "Answer: "
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

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
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
    )
    return llm