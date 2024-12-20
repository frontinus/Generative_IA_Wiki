from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from ollama import chat
from ollama import ChatResponse

# example of a simple rag pipeline

prompt = "What kind of things happenned in Prague during the 20th century?"


df = pd.read_csv("./historical_events_with_abstracts.csv")
# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed event labels
event_embeddings = model.encode(df['label'].tolist())
event_embeddings = np.array(event_embeddings, dtype='float32')

# Create FAISS index
index = faiss.IndexFlatL2(event_embeddings.shape[1])  # L2 similarity
index.add(event_embeddings)

# Save index
faiss.write_index(index, "historical_events_index.faiss")

def retrieve_events(query):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k=5)  # Retrieve top 5 events
    return df.iloc[indices[0]]  # Return corresponding events

def format_retrieved_documents(events_df):
    """
    Format the retrieved events DataFrame into a string suitable for context.
    """
    formatted_documents = ""
    for idx, row in events_df.iterrows():
        formatted_documents += f"- Event: {row['event']}\n  Label: {row['label']}\n  Date: {row['date']}\n  Abstract: {row['abstract']}\n\n"
    return formatted_documents


# Example query
rows = retrieve_events(prompt)
print(rows)
retrieved_documents = format_retrieved_documents(rows)

response: ChatResponse = chat(model='phi3:mini', messages=[
  {
    'role': 'user',
    'content': f"""Use the following documents to answer the question. Provide references to the document sources where applicable.

        Context:
        {retrieved_documents}

        Question:
        {prompt}

        Answer (with references):
        """,
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)