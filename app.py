# -*- coding: utf-8 -*-
"""Russian AI Assistant — Gradio Web App for Hugging Face Spaces"""

import os
import json
import time
import gradio as gr

from nlp_engine import SimpleNLP
from api_clients import AnthropicClient, OpenAIClient

nlp = SimpleNLP()

# ─── Config persistence ───
CONFIG_FILE = "config.json"

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return {"api_mode": "offline", "api_type": "anthropic", "api_key": ""}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

config = load_config()

# ─── Chat function ───
def respond(message, history):
    """Main chat function for Gradio ChatInterface."""
    if not message or not message.strip():
        return ""

    if config["api_mode"] == "online" and config["api_key"]:
        # Build history for API
        api_history = []
        for user_msg, bot_msg in history:
            api_history.append({"role": "user", "content": user_msg})
            api_history.append({"role": "assistant", "content": bot_msg})
        api_history.append({"role": "user", "content": message})

        if config["api_type"] == "anthropic":
            client = AnthropicClient(config["api_key"])
        else:
            client = OpenAIClient(config["api_key"])

        reply = client.send_message(api_history)
    else:
        time.sleep(0.3)
        reply = nlp.get_response(message) or "Извините, я не совсем понял."

    return reply


# ─── Settings UI logic ───
def save_settings(api_mode, api_type, api_key):
    config["api_mode"] = api_mode
    config["api_type"] = api_type
    config["api_key"] = api_key
    save_config(config)
    mode_display = "🌐 Онлайн" if api_mode == "online" else "💻 Офлайн"
    return f"✅ Настройки обновлены. Режим: {mode_display}."

def get_current_settings():
    return (
        config["api_mode"],
        config["api_type"],
        config["api_key"],
    )


# ─── Gradio UI ───
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    title="Русский AI Помощник",
    css="""
    .gradio-container { max-width: 800px !important; margin: auto; }
    .chat-message { font-size: 15px !important; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown("# 🤖 Русский AI Помощник")
    gr.Markdown("Ваш интеллектуальный собеседник на русском языке.")

    # ── Chat interface ──
    chatbot = gr.ChatInterface(
        fn=respond,
        title=None,
        description=None,
        placeholder="Введите ваше сообщение...",
        examples=[
            "Привет!",
            "Расскажи шутку",
            "Что ты умеешь?",
            "Напиши стихотворение",
            "Посоветуй хорошую книгу",
            "Посоветуй фильм на вечер",
            "Помоги придумать идею для проекта",
            "Расскажи интересный факт",
        ],
        clear_btn="🗑 Очистить диалог",
        submit_btn="➡",
        retry_btn=None,
        undo_btn=None,
    )

    # ── Settings accordion ──
    with gr.Accordion("⚙ Настройки", open=False):
        with gr.Row():
            api_mode = gr.Radio(
                choices=["offline", "online"],
                value=config["api_mode"],
                label="Режим работы",
                info="Offline — без интернета | Online — через API",
            )
        with gr.Row():
            api_type = gr.Radio(
                choices=["anthropic", "openai"],
                value=config["api_type"],
                label="Тип API",
                info="Anthropic (Claude) или OpenAI (ChatGPT)",
                visible=True,
            )
        api_key = gr.Textbox(
            label="API-ключ",
            placeholder="Введите ваш API-ключ...",
            value=config["api_key"],
            type="password",
        )
        save_btn = gr.Button("✅ Сохранить настройки", variant="primary")
        save_output = gr.Textbox(label="", show_label=False)

        save_btn.click(
            fn=save_settings,
            inputs=[api_mode, api_type, api_key],
            outputs=[save_output],
        )

        # Toggle API type visibility based on mode
        def toggle_api_type(mode):
            return gr.update(visible=(mode == "online"))

        api_mode.change(
            fn=toggle_api_type,
            inputs=api_mode,
            outputs=api_type,
        )

    gr.Markdown(
        "---\n"
        "💡 **Совет**: Для онлайн-режима требуется API-ключ.\n"
        "Получить: [console.anthropic.com](https://console.anthropic.com) "
        "или [platform.openai.com](https://platform.openai.com)"
    )

# ─── Launch ───
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
