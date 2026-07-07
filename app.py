# app.py – Normalized RelayFreeLLM Gateway
import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ===== API Keys =====
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# ===== Provider configurations =====
providers = []

# Mistral first (most reliable)
if MISTRAL_API_KEY:
    providers.append({
        'name': 'mistral',
        'url': 'https://api.mistral.ai/v1/chat/completions',
        'headers': {'Authorization': f'Bearer {MISTRAL_API_KEY}'},
        'model': 'mistral-small-latest',
        'format': 'openai'
    })

# DeepSeek
if DEEPSEEK_API_KEY:
    providers.append({
        'name': 'deepseek',
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'headers': {'Authorization': f'Bearer {DEEPSEEK_API_KEY}'},
        'model': 'deepseek-chat',
        'format': 'openai'
    })

# NVIDIA
if NVIDIA_API_KEY:
    providers.append({
        'name': 'nvidia',
        'url': 'https://integrate.api.nvidia.com/v1/chat/completions',
        'headers': {'Authorization': f'Bearer {NVIDIA_API_KEY}'},
        'model': 'meta/llama3-70b-instruct',
        'format': 'openai'
    })

# Groq
if GROQ_API_KEY:
    providers.append({
        'name': 'groq',
        'model': 'llama-3.3-70b-versatile',
        'url': 'https://api.groq.com/openai/v1/chat/completions',
        'headers': {'Authorization': f'Bearer {GROQ_API_KEY}'},
        'format': 'openai'
    })

# Gemini last (due to quota issues)
if GEMINI_API_KEY:
    providers.append({
        'name': 'gemini',
        'url': f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}',
        'format': 'gemini'
    })

def normalize_response(provider, raw_response):
    if provider['format'] == 'gemini':
        candidates = raw_response.get('candidates', [])
        if candidates:
            text = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        else:
            text = ''
        return {'choices': [{'message': {'content': text}, 'index': 0, 'finish_reason': 'stop'}]}
    else:
        return raw_response

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json()
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 400)
    temperature = data.get('temperature', 0.7)

    for provider in providers:
        try:
            print(f"🔄 Attempting provider: {provider['name']}")  # LOGGING
            headers = {'Content-Type': 'application/json'}
            if provider.get('headers'):
                headers.update(provider['headers'])

            if provider['format'] == 'gemini':
                user_content = None
                for msg in messages:
                    if msg['role'] == 'user':
                        user_content = msg['content']
                        break
                if not user_content:
                    continue
                payload = {
                    'contents': [{'parts': [{'text': user_content}]}],
                    'generationConfig': {
                        'maxOutputTokens': max_tokens,
                        'temperature': temperature
                    }
                }
            else:
                payload = {
                    'model': provider['model'],
                    'messages': messages,
                    'max_tokens': max_tokens,
                    'temperature': temperature
                }

            response = requests.post(
                provider['url'],
                json=payload,
                headers=headers,
                timeout=60
            )

            if response.status_code == 200:
                raw = response.json()
                # Check for error in response body
                if 'error' in raw:
                    error_msg = raw['error'].get('message', str(raw['error']))
                    print(f"Provider {provider['name']} returned error in body: {error_msg}")
                    continue
                normalized = normalize_response(provider, raw)
                if 'choices' in normalized and len(normalized['choices']) > 0:
                    content = normalized['choices'][0]['message'].get('content')
                    if content:
                        print(f"✅ Provider {provider['name']} succeeded")
                        return jsonify(normalized)
                    else:
                        print(f"Provider {provider['name']} returned empty content")
                        continue
                else:
                    print(f"Provider {provider['name']} invalid response: {raw}")
                    continue
            else:
                print(f"Provider {provider['name']} failed: {response.status_code} – {response.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"Provider {provider['name']} timed out")
            continue
        except Exception as e:
            print(f"Provider {provider['name']} error: {str(e)}")
            continue

    return jsonify({
        'error': 'All AI providers are currently unavailable. Please try again later.'
    }), 503

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'providers': len(providers),
        'active_providers': [p['name'] for p in providers]
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'RelayFreeLLM Gateway', 'status': 'running'})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)