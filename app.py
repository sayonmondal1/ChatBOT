from flask import Flask, render_template, request, Response
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextIteratorStreamer
import torch
import threading
app = Flask(__name__)
#Load model
MODEL_PATH = "./gpt2-alpaca-trained"  # change if needed
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
# Safety: GPT-2 has no pad token by default
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
# Use low memory on Windows/CPU; still fine on GPU
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)
model.eval()
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"[INFO] Using device: {device}")
def build_prompt(message: str) -> str:
    # Simple chat-style prompt
    return f"User: {message}\nBot:"
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/generate_stream")
def generate_stream():
    message = (request.args.get("message") or "").strip()
    if not message:
        return Response("event: error\ndata: Please provide a message.\n\n",
                        mimetype="text/event-stream")
    prompt = build_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Real-time streaming from transformers
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,           # Don't re-send the prompt
        skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=256,         # Adjust to taste
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    def sse():
        # Run generation in a background thread so we can iterate the streamer
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            for chunk in streamer:
                # Send incremental text chunks
                # Avoid newlines inside 'data:' by replacing with spaces for SSE
                safe_chunk = chunk.replace("\r", " ").replace("\n", " ")
                yield f"data: {safe_chunk}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        # Signal completion
        yield "event: done\ndata: [DONE]\n\n"
    return Response(sse(), mimetype="text/event-stream")
if __name__ == "__main__":
    # For local dev
    app.run(debug=True)