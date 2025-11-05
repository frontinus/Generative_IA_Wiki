from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from ollama import chat
from ollama import ChatResponse
import os
from openai import OpenAI
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = None#OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== LOAD MODEL AND DATA AT MODULE LEVEL =====
logger.info("Loading sentence transformer model...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

logger.info("Loading historical events data...")
DF = pd.read_csv("./historical_events_with_abstracts.csv")

# Combine label + abstract for richer embeddings
logger.info("Creating combined text embeddings...")
DF['combined_text'] = DF['label'] + " " + DF['abstract']

# Encode and convert to float32 (required by FAISS)
event_embeddings = MODEL.encode(DF['combined_text'].tolist(), show_progress_bar=True)
EVENT_EMBEDDINGS = np.array(event_embeddings, dtype='float32')

# Verify dtype
assert EVENT_EMBEDDINGS.dtype == np.float32, "FAISS requires float32 embeddings"

# Build FAISS index
logger.info("Building FAISS index...")
INDEX = faiss.IndexFlatL2(EVENT_EMBEDDINGS.shape[1])
INDEX.add(EVENT_EMBEDDINGS)

logger.info(f"✓ Loaded {len(DF)} events. Index ready with {EVENT_EMBEDDINGS.shape[1]} dimensions.")


# ===== HELPER FUNCTIONS =====
def format_retrieved_documents(events_df: pd.DataFrame) -> str:
    """
    Format the retrieved events DataFrame into a string suitable for context.
    
    Args:
        events_df: DataFrame containing retrieved events with columns:
                   ['event', 'label', 'date', 'abstract']
    
    Returns:
        Formatted string with document information
    """
    formatted_documents = "### Retrieved Documents:\n"
    for idx, row in events_df.iterrows():
        formatted_documents += f"""
Document {idx + 1}:
- **Event:** {row['event']}
- **Label:** {row['label']}
- **Date:** {row['date']}
- **Abstract:** {row['abstract']}

"""
    return formatted_documents


def retrieve_events(query: str, k: int = 5) -> pd.DataFrame:
    """
    Retrieve top-k most relevant events using FAISS similarity search.
    
    Args:
        query: User's search query
        k: Number of documents to retrieve (default: 5)
    
    Returns:
        DataFrame with top-k most relevant events
    
    Raises:
        ValueError: If query is empty or k is invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if k < 1 or k > len(DF):
        raise ValueError(f"k must be between 1 and {len(DF)}")
    
    # Encode query and ensure float32 dtype
    query_embedding = MODEL.encode([query])
    query_embedding = np.array(query_embedding, dtype='float32')
    
    # Verify dtype
    assert query_embedding.dtype == np.float32, "Query embedding must be float32"
    
    # Search FAISS index
    distances, indices = INDEX.search(query_embedding, k=k)
    
    logger.info(f"Retrieved {k} documents for query: '{query[:50]}...'")
    logger.debug(f"Distances: {distances[0]}")
    
    return DF.iloc[indices[0]]


def generate_answer(prompt: str, retrieved_docs: pd.DataFrame, use_openai: bool = False) -> str:
    """
    Generate answer using either OpenAI or local Ollama model.
    
    Args:
        prompt: User's question
        retrieved_docs: DataFrame with retrieved context documents
        use_openai: If True, use OpenAI GPT-4; otherwise use Ollama
    
    Returns:
        HTML-formatted answer string
    
    Raises:
        RuntimeError: If model fails to generate response
    """
    global client
    context = format_retrieved_documents(retrieved_docs)
    
    system_prompt = """
You are a highly knowledgeable assistant that answers questions based only on the provided context.
Do not introduce external information or speculate. Use the context strictly to construct your response.
If the context is insufficient to answer, respond by stating that explicitly.

CRITICAL FORMATTING RULES:
- Return ONLY raw HTML content
- Do NOT use markdown code blocks (```)
- Do NOT prefix with "html" or any other text
- Do NOT wrap in backticks
- Start directly with HTML tags like <p>, <div>, <ul>, etc.
- Format your answer as clean HTML that can be directly inserted into a <div> element

Example of correct response:
<p>World War II ended on September 2, 1945.</p>

Example of INCORRECT response (DO NOT DO THIS):
```html
<p>World War II ended on September 2, 1945.</p>
```
"""

    user_message = f"""
### Context:
{context}

### Question:
{prompt}

### Instructions:
Answer the question based on the context provided. Return your response as raw HTML content only.
Remember: No markdown, no code blocks, no prefix text - just pure HTML starting with a tag.
"""

    try:
        if use_openai:
            if client is None:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error("Cannot use OpenAI: OPENAI_API_KEY environment variable is not set.")
                    # We raise an error here so the user gets a proper 500 response
                    raise ValueError("Cannot use OpenAI: OPENAI_API_KEY environment variable is not set.")
                
                logger.info("Initializing OpenAI client...")
                client = OpenAI(api_key=api_key)
            
            logger.info("Generating answer with OpenAI GPT-4...")
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            answer = response.choices[0].message.content
            logger.info("✓ OpenAI response generated successfully")
            return answer
        else:
            logger.info("Generating answer with Ollama (phi3:mini)...")
            response: ChatResponse = chat(
                model='phi3:mini',
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': user_message,
                    },
                ]
            )
            answer = response.message.content
            logger.info("✓ Ollama response generated successfully")
            return answer
            
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise RuntimeError(f"Failed to generate answer: {str(e)}")


def rag(prompt: str, use_openai: bool = False, k: int = 5) -> str:
    """
    Perform Retrieval-Augmented Generation (RAG) on historical events.
    
    This function:
    1. Retrieves relevant historical events using semantic search
    2. Generates a contextual answer using either OpenAI or Ollama
    
    Args:
        prompt: User's question about historical events
        use_openai: If True, use OpenAI GPT-4; if False, use local Ollama (default: False)
        k: Number of documents to retrieve for context (default: 5)
    
    Returns:
        HTML-formatted answer based on retrieved documents
    
    Raises:
        ValueError: If prompt is empty
        RuntimeError: If retrieval or generation fails
    
    Example:
        >>> answer = rag("When did World War II end?")
        >>> print(answer)
        <p>World War II ended in 1945...</p>
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    logger.info(f"RAG query: '{prompt[:100]}...' (use_openai={use_openai})")
    
    try:
        # Step 1: Retrieve relevant documents
        docs = retrieve_events(prompt, k=k)
        
        # Step 2: Generate answer
        answer = generate_answer(prompt, docs, use_openai)
        
        logger.info("✓ RAG pipeline completed successfully")
        return answer
        
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise


# ===== OPTIONAL: EVALUATION FUNCTION =====
def evaluate_retrieval(query: str, expected_keywords: list, k: int = 5) -> dict:
    """
    Evaluate retrieval quality for a given query.
    
    Args:
        query: Search query
        expected_keywords: List of keywords that should appear in results
        k: Number of documents to retrieve
    
    Returns:
        Dictionary with evaluation metrics
    """
    docs = retrieve_events(query, k=k)
    
    # Check how many expected keywords appear in retrieved docs
    all_text = " ".join(docs['combined_text'].tolist()).lower()
    found_keywords = [kw for kw in expected_keywords if kw.lower() in all_text]
    
    return {
        'query': query,
        'retrieved_docs': len(docs),
        'expected_keywords': expected_keywords,
        'found_keywords': found_keywords,
        'precision': len(found_keywords) / len(expected_keywords) if expected_keywords else 0
    }


# ===== OPTIONAL: SAVE/LOAD INDEX =====
def save_index(path: str = "faiss_index.bin"):
    """Save FAISS index to disk for faster startup."""
    faiss.write_index(INDEX, path)
    logger.info(f"✓ FAISS index saved to {path}")


def load_index(path: str = "faiss_index.bin") -> Optional[faiss.Index]:
    """Load FAISS index from disk."""
    if os.path.exists(path):
        index = faiss.read_index(path)
        logger.info(f"✓ FAISS index loaded from {path}")
        return index
    return None


if __name__ == "__main__":
    # Test the pipeline
    print("\n" + "="*60)
    print("Testing RAG Pipeline")
    print("="*60 + "\n")
    
    test_queries = [
        "When did World War II end?",
        "Who invented the airplane?",
        "What happened during the French Revolution?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        try:
            answer = rag(query, use_openai=False)
            print(answer[:300] + "..." if len(answer) > 300 else answer)
            print()
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Optional: Save index for faster future loads
    # save_index()
    
    # Optional: Test retrieval evaluation
    print("\n" + "="*60)
    print("Testing Retrieval Evaluation")
    print("="*60 + "\n")
    
    eval_result = evaluate_retrieval(
        "World War II",
        expected_keywords=["war", "1945", "hitler", "allies"]
    )
    print(f"Evaluation: {eval_result}")
