import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load passages
def parse_wikimedia_dump(dump_file, num_passages=10000):
    """Parses the Wikimedia dump and returns passages."""
    passages = []
    with open(dump_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            passages.append(data["content"])
            if len(passages) >= num_passages:
                break
    return passages

# Create FAISS index
def build_faiss_index(passages, index_file="faiss_index"):
    vectorizer = TfidfVectorizer(max_features=768)
    embeddings = vectorizer.fit_transform(passages).toarray()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_file)
    return index, vectorizer

# Load fine-tuned model
def load_model_and_index(model_path, index_file, dump_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Load or build FAISS index
    passages = parse_wikimedia_dump(dump_file)
    try:
        index = faiss.read_index(index_file)
    except:
        index, vectorizer = build_faiss_index(passages, index_file)
    return model, tokenizer, index, passages, vectorizer
