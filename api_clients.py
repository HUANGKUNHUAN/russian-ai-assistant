# -*- coding: utf-8 -*-
"""API clients for Anthropic (Claude) and OpenAI"""

import requests


class AnthropicClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "claude-sonnet-4-6"

    def send_message(self, history: list[dict]) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        messages = [{"role": m["role"], "content": m["content"]}
                    for m in history if m["role"] in ("user", "assistant")]
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": "Вы — полезный, дружелюбный русскоязычный AI-ассистент. Отвечайте на русском языке.",
            "messages": messages,
        }
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages",
                                 headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["content"][0]["text"]
            return f"⚠️ Ошибка API: {resp.status_code}"
        except Exception as e:
            return f"⚠️ Ошибка: {str(e)}"


class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-4o-mini"

    def send_message(self, history: list[dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [{"role": "system", "content": "Вы — полезный, дружелюбный русскоязычный AI-ассистент. Отвечайте на русском языке."}]
        for m in history:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        try:
            resp = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers,
                                 json={"model": self.model, "messages": messages, "max_tokens": 2048},
                                 timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return f"⚠️ Ошибка API: {resp.status_code}"
        except Exception as e:
            return f"⚠️ Ошибка: {str(e)}"
