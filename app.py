# app.py - RelayFreeLLM Gateway
import os
import json
import requests
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ===== API Keys from environment variables =====
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# ===== Provider configurations =====
providers = []

# Gemini
if GEMINI_API_KEY:
    providers.append({
        'name': 'gemini',
        'model': 'gemini-2.0-flash',
        'url': f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}',
        'format': 'gemini'
    })

# Groq
if GROQ_API_KEY:
    providers.append({
        'name': 'groq',
        'model': 'mixtral-8x7b-32768',
        'url': 'https://api.groq.com/openai/v1/chat/completions',
        'headers': {'Authorization': f'Bearer {GROQ_API_KEY}'},
        'format': 'openai'
    })

# Mistral
if MISTRAL_API_KEY:
    providers.append({
        'name': 'mistral',
        'model': 'mistral-small-latest',
        'url': 'https://api.mistral.ai/v1/chat/completions',
        'headers': {'Authorization': f'Bearer {MISTRAL_API_KEY}'},
        'format': 'openai'
    })

# DeepSeek
if DEEPSEEK_API_KEY:
    providers.append({
        'name': 'deepseek',
        'model': 'deepseek-chat',
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'headers': {'Authorization': f'Bearer {DEEPSEEK_API_KEY}'},
        'format': 'openai'
    })

# NVIDIA
if NVIDIA_API_KEY:
    providers.append({
        'name': 'nvidia',
        'model': 'meta/llama3-70b-instruct',
        'url': 'https://integrate.api.nvidia.com/v1/chat/completions',
        'headers': {'Authorization': f'Bearer {NVIDIA_API_KEY}'},
        'format': 'openai'
    })

# ===== Format messages for different providers =====
def format_gemini_prompt(messages):
    user_content = None
    for msg in messages:
        if msg['role'] == 'user':
            user_content = msg['content']
            break
    return {'contents': [{'parts': [{'text': user_content or ''}]}]}

# ===== Main chat completion endpoint =====
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        model = data.get('model', 'gemini-2.0-flash')
        max_tokens = data.get('max_tokens', 400)
        temperature = data.get('temperature', 0.7)

        for provider in providers:
            try:
                headers = {'Content-Type': 'application/json'}
                if provider.get('headers'):
                    headers.update(provider['headers'])
                
                payload = {
                    'model': provider['model'],
                    'messages': messages,
                    'max_tokens': max_tokens,
                    'temperature': temperature
                }

                if provider['format'] == 'gemini':
                    payload = format_gemini_prompt(messages)
                    payload['generationConfig'] = {
                        'maxOutputTokens': max_tokens,
                        'temperature': temperature
                    }

                response = requests.post(
                    provider['url'],
                    json=payload,
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 200:
                    response_data = response.json()
                    
                    if provider['format'] == 'gemini':
                        content = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                        return jsonify({
                            'choices': [{
                                'message': {'content': content},
                                'index': 0,
                                'finish_reason': 'stop'
                            }],
                            'model': provider['name'],
                            'usage': {'total_tokens': len(content)}
                        })
                    else:
                        return jsonify(response_data)

                else:
                    print(f"Provider {provider['name']} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"Provider {provider['name']} error: {str(e)}")
                continue

        return jsonify({
            'error': 'All providers failed. Please try again later.'
        }), 503

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== Models endpoint =====
@app.route('/v1/models', methods=['GET'])
def list_models():
    model_names = [p['model'] for p in providers]
    return jsonify({
        'data': [{'id': name, 'object': 'model'} for name in model_names]
    })

# ===== Health check =====
@app.route('/health', methods=['GET'])
def health():
    provider_count = len(providers)
    return jsonify({
        'status': 'ok',
        'providers': provider_count,
        'active_providers': [p['name'] for p in providers]
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'RelayFreeLLM Gateway',
        'status': 'running',
        'providers': len(providers),
        'active': [p['name'] for p in providers]
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)