from setup_rag import load_model_and_index
from transformers import AutoTokenizer

# Load model and retriever
model_path = "./fine_tuned_model"
index_file = "faiss_index"
dump_file = "parsed_wikipedia.jsonl"
model, tokenizer, index, passages, vectorizer = load_model_and_index(model_path, index_file, dump_file)

# Query and generate
def query_rag_framework(query, index, vectorizer, passages, tokenizer, model, top_k=5):
    query_vector = vectorizer.transform([query]).toarray()
    _, indices = index.search(query_vector, top_k)
    retrieved_passages = [passages[i] for i in indices[0]]
    context = " ".join(retrieved_passages)
    input_text = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test query
query = "What is Artificial Intelligence?"
response = query_rag_framework(query, index, vectorizer, passages, tokenizer, model)
print(response)
