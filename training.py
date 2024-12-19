from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

model_name = "gpt2"  # Replace with your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)



# Example: Load preprocessed Wikipedia data in JSONL format
dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["content"], truncation=True, padding="longest", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)



training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")







