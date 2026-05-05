import gradio as gr
import requests

API_URL = "http://localhost:8000/query"

_WELCOME_MD = """
## Welcome to the Call Center Assistant!

I can help you with:
- **General Inquiries**: Answer questions about products, services, and policies
- **Account Support**: Help with account-related questions and information
- **Technical Assistance**: Guide you through technical or procedural issues
- **Information Lookup**: Retrieve relevant information from the knowledge base
"""

_ARCHITECTURE_MD = """
**Architecture:**
- Non-transactional: RAG + LLM
- Transactional: NER + LLM + API
- Response Analysis: XLM-T / GLiNER2
"""

_CSS = """
#send-btn { background-color: #5B5EA6; color: white; }
#header { text-align: center; }
"""


def _call_pipeline(dialog_history: list[str]) -> str:
    try:
        r = requests.post(API_URL, json={"dialog_history": dialog_history}, timeout=60)
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.ConnectionError:
        return "Error: could not reach the pipeline server. Is server.py running?"
    except Exception as e:
        return f"Error: {e}"


def send_message(message: str, chatbot: list, dialog_history: list[str]):
    message = message.strip()
    if not message:
        return chatbot, dialog_history, ""

    dialog_history = dialog_history + [message]
    response = _call_pipeline(dialog_history)
    dialog_history = dialog_history + [response]

    chatbot = chatbot + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return chatbot, dialog_history, ""


def clear_chat():
    return [], [], ""  # empty list is valid for both formats


with gr.Blocks(theme=gr.themes.Soft(), css=_CSS, title="Call Center Assistant") as demo:

    dialog_history = gr.State([])

    gr.Markdown(
        "# 🏢 Call Center Assistant — Powered by LLM + RAG + Advanced NLP",
        elem_id="header",
    )
    gr.Markdown(_WELCOME_MD)

    with gr.Row():

        # ── Left: chat ──────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Assistant")
            chatbot = gr.Chatbot(height=500, show_label=False)

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    scale=4,
                    show_label=True,
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1, elem_id="send-btn")

            clear_btn = gr.Button("Clear Chat 🗑️")

        # ── Right: monitoring ───────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 System Monitoring")
            gr.Markdown(_ARCHITECTURE_MD)

    send_btn.click(
        send_message,
        inputs=[msg_input, chatbot, dialog_history],
        outputs=[chatbot, dialog_history, msg_input],
    )
    msg_input.submit(
        send_message,
        inputs=[msg_input, chatbot, dialog_history],
        outputs=[chatbot, dialog_history, msg_input],
    )
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, dialog_history, msg_input],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
