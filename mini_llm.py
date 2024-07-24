from flask import Flask, request, jsonify
from huggingface_hub import login, InferenceClient
from pydantic import BaseModel
from dotenv import load_dotenv, set_key
import os
import re
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

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
        raise Exception("HF Token not set. Please provide it.")
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

def verify_app_token(request):
    token = request.headers.get('Authorization')
    if (token is None) or (token != APP_TOKEN):
        raise Exception("Invalid app token")

client = None

@app.route("/")
def hello():
    return """Welcome!,
              For Inference using Meta-Llama post to /inference
              For Inference using Lamini-Prompt post to /summarize"""

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "API is active"})

@app.route("/inference", methods=["POST"])
def run_inference():
    global client
    try:
        verify_app_token(request)
        
        data = request.get_json()
        inference_request = InferenceRequest(**data)

        text = preprocess_text(inference_request.text)
        
        if is_spam(text):
            return jsonify({"error": "Text is considered spam"}), 400

        if inference_request.hf_token:
            set_hf_token(inference_request.hf_token, inference_request.store_token)
            login(inference_request.hf_token, add_to_git_credential=True)

        hf_token = get_hf_token()

        if client is None or inference_request.hf_token:
            client = InferenceClient(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                token=hf_token,
            )

        messages = [{"role": "user", "content": text}]
        
        result = []
        for message in client.chat_completion(messages=messages, max_tokens=250, stream=True):
            result.append(message.choices[0].delta.content)

        return jsonify({"result": ''.join(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()