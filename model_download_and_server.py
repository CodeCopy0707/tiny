import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_DIR = "./tinyllama-1.1b-chat"

# Check if model is already downloaded
if not os.path.exists(LOCAL_DIR):
    print("Model not found locally. Downloading...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    os.makedirs(LOCAL_DIR, exist_ok=True)
    model.save_pretrained(LOCAL_DIR)
    tokenizer.save_pretrained(LOCAL_DIR)
    print("Model downloaded and saved locally.")
else:
    print("Model found locally. Loading...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_DIR,
        trust_remote_code=True
    )

# FastAPI Setup
app = FastAPI()

class RequestData(BaseModel):
    prompt: str
    max_new_tokens: int = 100

@app.post("/generate")
def generate_text(data: RequestData):
    inputs = tokenizer(data.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=data.max_new_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
