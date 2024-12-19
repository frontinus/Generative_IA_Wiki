from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Initialize a pipeline for text generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Evaluate with a sample prompt
prompt = "What is Artificial Intelligence?"
outputs = generator(prompt, max_length=100, num_return_sequences=1)

# Print the result
print("Generated Output:")
for i, output in enumerate(outputs):
    print(f"{i+1}: {output['generated_text']}")
