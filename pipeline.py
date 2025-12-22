from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from ollama import chat
from ollama import ChatResponse
import os
from openai import OpenAI
import logging
from typing import Optional, Tuple, List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 data_path: str = "./historical_events_with_abstracts.csv", 
                 model_name: str = 'all-MiniLM-L6-v2',
                 index_path: str = "faiss_index.bin"):
        """
        Initialize the RAG Pipeline.
        
        Args:
            data_path: Path to the CSV data file.
            model_name: Name of the sentence-transformer model.
            index_path: Path to save/load the FAISS index.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.index_path = index_path
        
        self.model = None
        self.df = None
        self.index = None
        self.openai_client = None
        
        # Lazy load resources
        self._load_resources()

    def _load_resources(self):
        """Load model, data, and index."""
        try:
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name)
            
            logger.info("Loading historical events data...")
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
            self.df = pd.read_csv(self.data_path)
            
            # Combine label + abstract for richer embeddings if not already present
            if 'combined_text' not in self.df.columns:
                logger.info("Creating combined text embeddings...")
                self.df['combined_text'] = self.df['label'] + " " + self.df['abstract']
            
            # Load or build index
            if os.path.exists(self.index_path):
                logger.info(f"Loading FAISS index from {self.index_path}...")
                self.index = faiss.read_index(self.index_path)
            else:
                logger.info("Building FAISS index...")
                self._build_index()
                
            logger.info(f"✓ Pipeline ready. Loaded {len(self.df)} events.")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise

    def _build_index(self):
        """Build FAISS index from data."""
        if self.model is None or self.df is None:
            raise RuntimeError("Model or Data not loaded")
            
        # Encode and convert to float32 (required by FAISS)
        event_embeddings = self.model.encode(self.df['combined_text'].tolist(), show_progress_bar=True)
        event_embeddings = np.array(event_embeddings, dtype='float32')
        
        # Verify dtype
        if event_embeddings.dtype != np.float32:
            event_embeddings = event_embeddings.astype(np.float32)
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(event_embeddings.shape[1])
        self.index.add(event_embeddings)
        
        # Save index
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"✓ FAISS index saved to {self.index_path}")
        except Exception as e:
            logger.warning(f"Could not save FAISS index: {e}")

    def _get_openai_client(self) -> OpenAI:
        """Get or initialize OpenAI client."""
        if self.openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Cannot use OpenAI: OPENAI_API_KEY environment variable is not set.")
            self.openai_client = OpenAI(api_key=api_key)
        return self.openai_client

    def format_retrieved_documents(self, events_df: pd.DataFrame) -> str:
        """Format retrieved documents for context."""
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

    def retrieve_events(self, query: str, k: int = 5) -> pd.DataFrame:
        """Retrieve top-k most relevant events."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if k < 1:
            k = 1
        if k > len(self.df):
            k = len(self.df)
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype='float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k=k)
        
        logger.info(f"Retrieved {k} documents for query: '{query[:50]}...'")
        return self.df.iloc[indices[0]]

    def generate_answer(self, prompt: str, retrieved_docs: pd.DataFrame, use_openai: bool = False) -> str:
        """Generate answer using LLM."""
        context = self.format_retrieved_documents(retrieved_docs)
        
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
- Use <strong> for emphasis where appropriate.

Example of correct response:
<p>World War II ended on <strong>September 2, 1945</strong>.</p>

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
                client = self._get_openai_client()
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
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_message},
                    ]
                )
                answer = response.message.content
                logger.info("✓ Ollama response generated successfully")
                return answer
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise RuntimeError(f"Failed to generate answer: {str(e)}")

    def run(self, prompt: str, use_openai: bool = False, k: int = 5) -> str:
        """Run the full RAG pipeline."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        logger.info(f"RAG query: '{prompt[:100]}...' (use_openai={use_openai})")
        
        try:
            # Step 1: Retrieve relevant documents
            docs = self.retrieve_events(prompt, k=k)
            
            # Step 2: Generate answer
            answer = self.generate_answer(prompt, docs, use_openai)
            
            logger.info("✓ RAG pipeline completed successfully")
            return answer
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_documents": len(self.df) if self.df is not None else 0,
            "embedding_dimensions": self.index.d if self.index is not None else 0,
            "index_size": self.index.ntotal if self.index is not None else 0,
            "model": self.model_name,
            "backends": ["ollama (phi3:mini)", "openai (gpt-4-turbo)"]
        }

# Singleton instance for backward compatibility if needed, 
# but preferably instantiated in app.py
# rag_pipeline = RAGPipeline() 

if __name__ == "__main__":
    # Test the pipeline
    print("\n" + "="*60)
    print("Testing RAG Pipeline")
    print("="*60 + "\n")
    
    try:
        pipeline = RAGPipeline()
        
        test_queries = [
            "When did World War II end?",
            "Who invented the airplane?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 60)
            
            try:
                answer = pipeline.run(query, use_openai=False)
                print(answer[:300] + "..." if len(answer) > 300 else answer)
                print()
            except Exception as e:
                print(f"Error: {e}\n")
                
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
