from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import torch
# Load & Format JSON Dataset
dataset = load_dataset("json", data_files="./dataset/alpaca_data.json")
#dataset = load_dataset("json", data_files=uploaded)
def format_prompt(example):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output = example["output"].strip()
    if input_text:
        prompt = f"{instruction}\n\nInput:\n{input_text}\n\nResponse:\n{output}"
    else:
        prompt = f"{instruction}\n\nResponse:\n{output}"
    return {"text": prompt}
dataset = dataset["train"].map(format_prompt)
# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Required for GPT-2
def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Load Model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Training arguments with FP16 enabled
training_args = TrainingArguments(
    output_dir="./gpt2-alpaca",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    overwrite_output_dir=True,
    fp16=torch.cuda.is_available()  # Use FP16 if GPU is available
)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)
# Train
trainer.train()
# Save model
trainer.save_model("./gpt2-alpaca-trained")
tokenizer.save_pretrained("./gpt2-alpaca-trained")