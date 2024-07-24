from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv, set_key
from typing import List
from huggingface_hub import login, InferenceClient
import os
import re

# Load environment variables
load_dotenv()

app = FastAPI()

APP_TOKEN = os.getenv("APP_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

def set_hf_token(token: str, store: bool):
    global HF_TOKEN
    HF_TOKEN = token
    if store:
        set_key(".env", "HF_TOKEN", token)

def get_hf_token():
    global HF_TOKEN
    if not HF_TOKEN:
        raise HTTPException(status_code=403, detail="HF Token not set. Please provide it.")
    return HF_TOKEN

# Log in to Hugging Face if token is already set
if HF_TOKEN:
    login(HF_TOKEN, add_to_git_credential=True)

class InferenceRequest(BaseModel):
    text: str
    hf_token: str = None
    store_token: bool = False

def preprocess_text(text: str) -> str:
    # Basic preprocessing: lowercasing, removing extra spaces
    return re.sub(r'\s+', ' ', text.strip().lower())

def is_spam(text: str) -> bool:
    # Check if text contains gibberish or is too short
    if len(text) < 15 or re.search(r'(.)\1{4,}', text):
        return True
    return False

async def verify_token(request: Request):
    token = request.headers.get('Authorization')
    if token != APP_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# Load the client once
client = None

@app.get("/")
async def hello():
    return """Welcome!,
                For Inference using Meta-Llama post to /inference
                For Inference using Lamini-Prompt post to /summarize"""

@app.get("/ping")
async def ping():
    return {"message": "API is active"}

@app.post("/inference", dependencies=[Depends(verify_token)])
async def run_inference(request: InferenceRequest):
    global client
    try:
        text = preprocess_text(request.text)
        
        if is_spam(text):
            raise HTTPException(status_code=400, detail="Text is considered spam")

        if request.hf_token:
            set_hf_token(request.hf_token, request.store_token)
            login(request.hf_token, add_to_git_credential=True)

        hf_token = get_hf_token()
        
        if client is None or request.hf_token:
            client = InferenceClient(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                token=hf_token,
            )

        messages = [{"role": "user", "content": text}]
        
        result = []
        async for message in client.chat_completion(messages=messages, max_tokens=250, stream=True):
            result.append(message.choices[0].delta.content)

        return {"result": ''.join(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
