import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Path to your fine-tuned model
MODEL_PATH = "./gpt2-alpaca-trained"
# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
# Load model
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)
model.to(device)
model.eval()
print("\n✅ Model loaded successfully. Type 'exit' to quit.\n")
while True:
    instruction = input("Instruction: ").strip()
    if instruction.lower() == "exit":
        break
    input_text = input("Input (optional): ").strip()
    if input_text:
        prompt = f"{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"
    else:
        prompt = f"{instruction}\n\nResponse:\n"
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the new part of the response
    response = generated_text[len(prompt):].strip()
    print(f"\n[Response]: {response}\n")