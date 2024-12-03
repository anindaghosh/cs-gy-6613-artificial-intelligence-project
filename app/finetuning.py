from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Load Dataset
dataset = load_dataset(
    "json", data_files={"train": "train.jsonl", "test": "test.jsonl"}
)

# Load Pretrained Model and Tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Tokenize Dataset
def preprocess_data(batch):
    inputs = batch["instruction"]
    targets = batch["response"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_data = dataset.map(preprocess_data, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Use mixed precision for speed (if GPU supports it)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
)

# Train the Model
trainer.train()

# Save Fine-Tuned Model
model.save_pretrained("./fine_tuned_t5")
tokenizer.save_pretrained("./fine_tuned_t5")
