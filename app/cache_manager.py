from langchain_redis import RedisCache
from langchain_core.globals import set_llm_cache


def setup_redis_cache(redis_url="redis://localhost:6379"):
    """
    Set up Redis caching for the chatbot.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    semantic_cache = RedisSemanticCache(
        embeddings=embeddings,
        redis_url="redis://localhost:6379",
        distance_threshold=0.1
    )
    set_llm_cache(semantic_cache)
    return semantic_cache

    

    

