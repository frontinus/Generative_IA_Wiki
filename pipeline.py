from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from ollama import chat
from ollama import ChatResponse
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# example of a simple rag pipeline

def rag(prompt):
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
        Format the retrieved events DataFrame into a string suitable for context,
        clearly marking it as context.
        """
        formatted_documents = "### Retrieved Documents:\n"
        for idx, row in events_df.iterrows():
            formatted_documents += f"""
            Document {idx + 1}:
            - **Event:** {row['event']}
            - **Label:** {row['label']}
            - **Date:** {row['date']}
            - **Abstract:** {row['abstract']}
            \n"""
        return formatted_documents



    # Example query
    rows = retrieve_events(prompt)
    print(rows)
    retrieved_documents = format_retrieved_documents(rows)

    sent = f"""
            ### Context:
            {retrieved_documents}

            ### Question:
            {prompt}

            ### Answer:
            Please format your response in HTML.
        """
    print(sent)
    response: ChatResponse = chat(model='phi3:mini', messages=[
        {
        'role': 'system',
        'content': """
            You are a highly knowledgeable assistant that answers questions based only on the provided context.
            Do not introduce external information or speculate. Use the context strictly to construct your response.
            If the context is insufficient to answer, respond by stating that explicitly.
            Format your answers in HTML when responding to the user.
        """,
    },
    {
        'role': 'user',
        'content': f"""
            ### Context:
            {retrieved_documents}

            ### Question:
            {prompt}

            ### Answer:
            Please format your response in HTML.
        """,
    },
    ])
    return response.message.content

def rag_openai(prompt):


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
        Format the retrieved events DataFrame into a string suitable for context,
        clearly marking it as context.
        """
        formatted_documents = "### Retrieved Documents:\n"
        for idx, row in events_df.iterrows():
            formatted_documents += f"""
            Document {idx + 1}:
            - **Event:** {row['event']}
            - **Label:** {row['label']}
            - **Date:** {row['date']}
            - **Abstract:** {row['abstract']}
            \n"""
        return formatted_documents



    # Example query
    rows = retrieve_events(prompt)
    print(rows)
    retrieved_documents = format_retrieved_documents(rows)


    response = client.chat.completions.create(model="gpt-4-turbo",  # Use "gpt-4" for the full model or "gpt-4-turbo" for a lighter version
    messages=[
        {"role": "system", "content": """
            You are a highly knowledgeable assistant that answers questions based only on the provided context.
            Do not introduce external information or speculate. Use the context strictly to construct your response.
            If the context is insufficient to answer, respond by stating that explicitly.
            Format your answers in HTML when responding to the user so that it is the child of a <div>.
        """},
        {"role": "user", "content": f"""
            ### Context:
            {retrieved_documents}

            ### Question:
            {prompt}

            ### Answer:
            Please format your response in HTML that could be the child of a <div>.
        """}
    ],
    max_tokens=1000,
    temperature=0.7)  # Adjust temperature for creativity (lower = more deterministic))
    return response.choices[0].message.content

